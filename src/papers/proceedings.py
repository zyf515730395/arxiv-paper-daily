"""Collect official proceedings and match accepted papers without model inference.

Run ``python -m papers.proceedings --apply`` to refresh the public catalog.
Without --apply only a private preview and coverage report are written.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import html
import json
from pathlib import Path
import re
from threading import Lock
import unicodedata
from urllib.parse import urlencode, urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

from papers.paths import ROOT
from shared.rendering import atomic_write_text

CATALOG = ROOT / 'content/papers/conference-papers.json'


def normalize_title(title: str) -> str:
    text = unicodedata.normalize('NFKC', html.unescape(title)).casefold()
    return ''.join(char for char in text if char.isalnum())


def paper_source_version(paper: dict) -> str:
    """Version the official identity fields that can change downstream evidence."""
    fields = {key: paper[key] for key in ('title', 'url', 'doi', 'arxiv_id', 'published')
              if paper.get(key)}
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def load_catalog(path: Path = CATALOG) -> dict:
    if not path.exists():
        return {'version': 1, 'editions': []}
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 1 or not isinstance(data.get('editions'), list):
        raise ValueError('Invalid proceedings catalog')
    editions = set()
    for edition in data['editions']:
        if not isinstance(edition.get('edition'), str) or edition['edition'] in editions:
            raise ValueError('Invalid or duplicate conference edition')
        if edition.get('source_version') is not None and not re.fullmatch(
                r'[0-9a-f]{64}', str(edition['source_version'])):
            raise ValueError('Invalid conference source version')
        editions.add(edition['edition'])
        for paper in edition['papers']:
            if not paper.get('title') or urlparse(paper.get('url', '')).scheme != 'https':
                raise ValueError('Invalid accepted-paper title or URL')
            if paper.get('source_version') is not None and not re.fullmatch(
                    r'[0-9a-f]{64}', str(paper['source_version'])):
                raise ValueError('Invalid paper source version')
    return data


def match_papers(rows: list[dict], catalog: dict) -> dict[str, list[dict]]:
    """Only unique exact normalized titles or explicit arXiv IDs are eligible."""
    by_title = defaultdict(set)
    ids = {row['id'] for row in rows}
    for row in rows:
        by_title[normalize_title(row['title'])].add(row['id'])
    found = defaultdict(dict)
    for edition in catalog['editions']:
        for paper in edition['papers']:
            explicit_id = re.sub(r'v\d+$', '', paper.get('arxiv_id', ''))
            candidates = {explicit_id} if explicit_id in ids else by_title.get(normalize_title(paper['title']), set())
            if len(candidates) != 1:
                continue
            paper_id = next(iter(candidates))
            found[paper_id][edition['edition']] = {
                'edition': edition['edition'], 'url': paper['url'],
                'match': 'arxiv_id' if explicit_id in ids else 'normalized_title',
            }
    return {paper_id: [editions[key] for key in sorted(editions)] for paper_id, editions in found.items()}


def parse_papers(text: str, url: str) -> list[dict]:
    """Source-specific selectors exclude navigation, workshops and submissions."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    host = urlparse(url).hostname or ''
    records = []

    def add(title, link):
        title = ' '.join(title.split())
        link = urljoin(url, html.unescape(link)).replace('http://', 'https://', 1)
        if title and urlparse(link).scheme == 'https':
            records.append({'title': title, 'url': link})

    if host == 'proceedings.mlr.press':
        for block in soup.select('div.paper'):
            title, link = block.select_one('.title'), block.select_one('a[href$=".html"]')
            if title and link:
                add(title.get_text(' ', strip=True), link['href'])
    elif host == 'openaccess.thecvf.com' or host in {'ecva.net', 'www.ecva.net'}:
        for link in soup.select('dt.ptitle a[href]'):
            add(link.get_text(' ', strip=True), link['href'])
    elif host in {'iclr.cc', 'icml.cc', 'neurips.cc', 'eccv.ecva.net'}:
        year = re.search(r'/virtual/(\d{4})/', url)
        if year:
            for link in soup.select('a[href]'):
                if re.search(rf'/virtual/{year[1]}/(?:poster|oral|spotlight)/\d+', link['href']):
                    add(link.get_text(' ', strip=True), link['href'])
    elif host == 'papers.nips.cc':
        for link in soup.select('a[href]'):
            if re.search(r'/hash/.+-Abstract(?:-[A-Za-z_]+)?\.html$', link['href']):
                add(link.get_text(' ', strip=True), link['href'])
    elif host == 'ojs.aaai.org':
        for link in soup.select('.obj_article_summary .title a[href]'):
            add(link.get_text(' ', strip=True), link['href'])
    elif 'conference-program.org' in host or 'conference-schedule.org' in host:
        for row in soup.select('tr[ssid]'):
            if not re.match(r'(?:papers|paper|techpap|tp)_', row.get('ssid','')):
                continue
            title = row.select_one('.presentation-title, .title-speakers-td')
            if title:
                link = title.select_one('a[href]')
                if link:
                    add(link.get_text(' ',strip=True), link['href'])
                else:
                    name = ' '.join(str(node).strip() for node in title.contents if isinstance(node,str))
                    add(name, urljoin(url, '/presentation/') + '?id=' + row['ssid'] + '&sess=' + row.get('psid',''))
    elif host == 'ras.papercept.net':
        for title in soup.select('.pTtl'):
            link = title.select_one('a[onclick]')
            identity = re.search(r"viewAbstract\('?(\d+)'?\)", link.get('onclick','')) if link else None
            if identity:
                add(title.get_text(' ',strip=True),url+'#'+identity[1])
    return list({(normalize_title(p['title']), p['url']): p for p in records}.values())


class Fetcher:
    def __init__(self, cache: Path, refresh: bool = False):
        self.cache, self.refresh = cache, refresh
        self._accessed = {}
        self._access_lock = Lock()
        cache.mkdir(parents=True, exist_ok=True)

    def reset_provenance(self) -> None:
        with self._access_lock:
            self._accessed.clear()

    def provenance(self) -> list[dict]:
        with self._access_lock:
            return [dict(value) for _, value in sorted(self._accessed.items())]

    def _record(self, url: str, sha256: str, cache_status: str) -> None:
        with self._access_lock:
            self._accessed[url] = {
                'url': url, 'sha256': sha256, 'cache_status': cache_status,
            }

    def get(self, url: str) -> str:
        path = self.cache / (hashlib.sha256(url.encode()).hexdigest() + '.html')
        ok_path, metadata_path = path.with_suffix('.ok'), path.with_suffix('.json')
        cached, metadata = None, {}
        try:
            if ok_path.exists():
                cached = path.read_text(encoding='utf-8')
                metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
                digest = hashlib.sha256(cached.encode()).hexdigest()
                if metadata and (metadata.get('url') != url or metadata.get('sha256') != digest):
                    cached, metadata = None, {}
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            cached, metadata = None, {}
        if cached is not None and not self.refresh:
            self._record(url, hashlib.sha256(cached.encode()).hexdigest(), 'cached')
            return cached
        headers = {'User-Agent':'LOKEN-Proceedings/1.0'}
        if cached is not None:
            if metadata.get('etag'):
                headers['If-None-Match'] = str(metadata['etag'])
            if metadata.get('last_modified'):
                headers['If-Modified-Since'] = str(metadata['last_modified'])
        with requests.Session() as session:
            session.mount('https://', HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
            response = session.get(url, timeout=45, headers=headers)
        if response.status_code == 304 and cached is not None:
            digest = hashlib.sha256(cached.encode()).hexdigest()
            receipt = {
                'version': 1, 'url': url, 'sha256': digest,
                'etag': response.headers.get('ETag') or metadata.get('etag'),
                'last_modified': response.headers.get('Last-Modified') or metadata.get('last_modified'),
                'revalidated_on': date.today().isoformat(),
            }
            atomic_write_text(metadata_path, json.dumps(receipt, ensure_ascii=False, sort_keys=True) + '\n')
            atomic_write_text(ok_path, date.today().isoformat())
            self._record(url, digest, 'revalidated')
            return cached
        response.raise_for_status()
        response.encoding = 'windows-1252' if 'charset=windows-1252' in response.text[:1000] else 'utf-8'
        text = response.text
        digest = hashlib.sha256(text.encode()).hexdigest()
        old_digest = hashlib.sha256(cached.encode()).hexdigest() if cached is not None else None
        atomic_write_text(path, text)
        atomic_write_text(ok_path, date.today().isoformat())
        receipt = {
            'version': 1, 'url': url, 'sha256': digest,
            'etag': response.headers.get('ETag'),
            'last_modified': response.headers.get('Last-Modified'),
            'fetched_on': date.today().isoformat(),
        }
        atomic_write_text(metadata_path, json.dumps(receipt, ensure_ascii=False, sort_keys=True) + '\n')
        self._record(url, digest, 'unchanged' if old_digest == digest else 'changed')
        return text


def collect_edition(meeting: dict, fetch: Fetcher) -> list[dict]:
    from bs4 import BeautifulSoup
    if meeting.get('crossref_sources'):
        return collect_crossref(meeting['crossref_sources'], fetch)
    sources = meeting.get('accepted_papers_urls') or [meeting.get('proceedings_url')]
    papers = []
    for url in sources:
        if not url:
            continue
        if urlparse(url).hostname == 'openaccess.thecvf.com' and '?' not in url:
            url += '?day=all'
        text = fetch.get(url)
        soup = BeautifulSoup(text, 'html.parser')
        children = []
        if urlparse(url).hostname == 'papers.nips.cc':
            children = [urljoin(url,a['href']) for a in soup.select('a[href]') if 'main-conference' in a['href']]
        elif urlparse(url).hostname == 'aaai.org':
            children = [a['href'] for a in soup.select('a[href]') if 'ojs.aaai.org/index.php/AAAI/issue/view/' in a['href'] and 'Technical Track' in a.get_text()]
        elif urlparse(url).hostname == 'ras.papercept.net':
            children = [urljoin(url,a['href']) for a in soup.select('a[href]') if re.search(r'_ContentListWeb_\d+\.html$',a['href'])]
        elif 'conference-program.org' in url or 'conference-schedule.org' in url:
            children = [urljoin(url,node['source']) for node in soup.select('.tablesched[source]')]
            if not children and not soup.select('tr[ssid^="papers_"]'):
                children = [urljoin(url,a['href']) for row in soup.select('tr[etypes]') if 'Technical Paper' in row.get_text() for a in row.select('.presentation-title a[href]') if 'sess=' in a['href']]
            children = list(dict.fromkeys(children))
        papers.extend(parse_papers(text,url))
        # A child failure aborts the edition so a partial list cannot replace it.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for child, body in zip(children, pool.map(fetch.get, children)):
                papers.extend(parse_papers(body,child))
    return sorted({(normalize_title(p['title']),p['url']):p for p in papers}.values(),key=lambda p:p['title'].casefold())


def collect_crossref(sources: list[dict], fetch: Fetcher) -> list[dict]:
    """Use publisher-deposited metadata, exact container filters and all pages."""
    from bs4 import BeautifulSoup
    papers = {}
    for source in sources:
        offset = 0
        while True:
            url = 'https://api.crossref.org/works?' + urlencode({
                'filter':source['filter'], 'rows':1000, 'offset':offset,
                'select':'title,DOI,container-title,volume,issue,type',
            })
            message = json.loads(fetch.get(url))['message']
            items = message['items']
            if not items and offset < message['total-results']:
                raise ValueError('Crossref pagination ended before total-results')
            for item in items:
                if any(str(item.get(key,'')) != str(source[key]) for key in ('volume','issue') if key in source):
                    continue
                if not item.get('title') or not item.get('DOI'):
                    continue
                title = BeautifulSoup(item['title'][0],'html.parser').get_text(' ',strip=True)
                if title.lower() in {'front matter','back matter','table of contents','title page','index','author index'}:
                    continue
                papers[item['DOI']] = {'title':title,'url':'https://doi.org/'+item['DOI']}
            offset += len(items)
            if offset >= message['total-results']:
                break
            if offset >= 10000:
                raise ValueError('Proceedings filter unexpectedly exceeds 10000 records')
    return sorted(papers.values(),key=lambda p:p['title'].casefold())


def refresh_catalog(config: dict, previous: dict, fetch: Fetcher, today: date, only: set[str] | None = None) -> tuple[dict, dict]:
    old = {e['edition']: e for e in previous['editions']}
    editions, report = [], []
    for conference in config['conferences']:
        if not conference.get('enabled', True):
            continue
        for meeting in conference['meetings']:
            name = meeting['edition']
            year = int(re.search(r'\d{4}',name)[0])
            if year < 2024:
                continue
            if only and name not in only:
                if name in old:
                    editions.append(old[name])
                continue
            entry = dict(old.get(name, {'edition':name,'papers':[]}))
            entry.update(conference=conference['id'], source_url=meeting.get('proceedings_url'), checked_on=today.isoformat())
            if meeting.get('crossref_sources'):
                entry['collection_sources'] = meeting['crossref_sources']
            elif meeting.get('accepted_papers_urls'):
                entry['collection_sources'] = meeting['accepted_papers_urls']
            end = str(meeting.get('end_date',''))
            entry['meeting_status'] = 'held' if end and end < today.isoformat() else 'upcoming_or_dates_pending'
            fetch.reset_provenance()
            try:
                if not meeting.get('proceedings_url') and not meeting.get('accepted_papers_urls'):
                    entry['status'] = 'pending'
                else:
                    papers = collect_edition(meeting,fetch)
                    if not papers:
                        raise ValueError('No accepted-paper entries parsed; source requires inspection')
                    if entry['papers'] and len(papers) < len(entry['papers']):
                        raise ValueError('List shrank; retain previous version pending review')
                    papers = [{**paper, 'source_version': paper_source_version(paper)} for paper in papers]
                    provenance = fetch.provenance()
                    source_version = hashlib.sha256(json.dumps(
                        [{k: source[k] for k in ('url', 'sha256')} for source in provenance],
                        sort_keys=True, separators=(',', ':'),
                    ).encode()).hexdigest()
                    previous_version = entry.get('source_version')
                    entry.update(papers=papers,status='collected',collected_on=today.isoformat(),
                                 source_version=source_version)
                report.append({'edition':name,'status':entry['status'],'papers':len(entry['papers']),
                               'source_version':entry.get('source_version'),
                               'source_changed':bool(previous_version and previous_version != entry.get('source_version'))
                               if entry['status'] == 'collected' else False})
            except (requests.RequestException, ValueError) as error:
                entry['status'] = 'refresh_failed' if entry['papers'] else 'unavailable'
                report.append({'edition':name,'status':entry['status'],'papers':len(entry['papers']),'error':str(error)})
            editions.append(entry)
            print(f"Checked {name}: {entry['status']} ({len(entry['papers'])})", flush=True)
    return {'version':1,'editions':editions}, {'checked_on':today.isoformat(),'editions':report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--refresh',action='store_true',help='Conditionally revalidate cached successful sources')
    parser.add_argument('--edition',action='append',help='Retry only this configured edition; repeatable')
    args = parser.parse_args(argv)
    config = yaml.safe_load((ROOT/'config/conferences.yaml').read_text(encoding='utf-8'))
    from papers.conferences import load_conferences
    load_conferences(ROOT/'config/conferences.yaml')
    cache = ROOT/'build/conferences'
    before = CATALOG.read_bytes() if CATALOG.exists() else None
    known = {m['edition'] for c in config['conferences'] for m in c['meetings']}
    if args.edition and set(args.edition) - known:
        parser.error('Unknown conference edition')
    catalog, report = refresh_catalog(config,load_catalog(),Fetcher(cache,args.refresh),date.today(),set(args.edition or []))
    from papers.site import parse_entry
    archive = json.loads((ROOT/'content/papers/archive.json').read_text(encoding='utf-8'))
    rows = {pid:parse_entry(pid,entry) for entries in archive.values() for pid,entry in entries.items()}
    matches = match_papers(list(rows.values()),catalog)
    report.update(archive_papers=len(rows),matched_papers=len(matches),matches=matches)
    atomic_write_text(cache/'report.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    preview = cache/'catalog-preview.json'
    atomic_write_text(preview,json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    load_catalog(preview)
    if args.apply:
        if (CATALOG.read_bytes() if CATALOG.exists() else None) != before:
            raise RuntimeError('Catalog changed during collection; retry without overwriting it')
        atomic_write_text(CATALOG,preview.read_text(encoding='utf-8'))
    for entry in report['editions']:
        print(f"{entry['edition']}: {entry['status']} ({entry['papers']})")
    print(f'Matched {len(matches)}/{len(rows)} archive papers')
    return 3 if any(e['status'] in {'unavailable','refresh_failed'} for e in report['editions']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
