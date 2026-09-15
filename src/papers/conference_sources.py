"""Bounded, identity-checked official sources for conference library summaries.

Unlike the arXiv queue, conference IDs never masquerade as arXiv identifiers.
All downloads and evidence receipts belong to the caller's private source cache.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote, unquote, urljoin, urlsplit

import requests

from .proceedings import normalize_title
from .summaries.acquisition import _atomic_write
from .summaries.extraction import extract_pdf_document, extract_introduction, _validate_document
from .summaries.models import AcquiredPaper, PaperDocument, PaperSection, PaperSummaryError


HOSTS = frozenset({
    'openaccess.thecvf.com', 'proceedings.mlr.press', 'papers.nips.cc',
    'papers.neurips.cc', 'icml.cc', 'iclr.cc', 'neurips.cc', 'eccv.ecva.net',
    'ecva.net', 'www.ecva.net', 'ojs.aaai.org', 'aaai.org',
    'openreview.net', 'arxiv.org', 'dl.acm.org', 'ieeexplore.ieee.org',
    'doi.org', 'api.crossref.org', 'proceedings.iclr.cc',
})
# Explicit publisher transitions; a redirect to another otherwise supported
# publisher is still rejected unless this pair is declared here.
REDIRECT_PAIRS = frozenset({
    ('papers.nips.cc', 'papers.neurips.cc'), ('papers.neurips.cc', 'papers.nips.cc'),
    ('ecva.net', 'www.ecva.net'), ('www.ecva.net', 'ecva.net'),
    ('doi.org', 'dl.acm.org'), ('doi.org', 'ieeexplore.ieee.org'),
})
MAX_BYTES = 64 * 1024 * 1024
# Verified from ECCV 2026's official per-paper pages. This CDN is deliberately
# absent from HOSTS: only an exact PDF link from the matching edition is usable.
_EVENTHOSTS_PDF = re.compile(r'https://media\.eventhosts\.cc/Conferences/ECCV2026/pdfs/[0-9]+\.pdf')


def _official_cdn_link(url: str, page_url: str) -> bool:
    return bool(_EVENTHOSTS_PDF.fullmatch(url) and re.fullmatch(
        r'https://eccv\.ecva\.net/virtual/2026/(?:poster|oral|spotlight)/[0-9]+', page_url))


def _safe_url(url: str, *, verified_cdn_urls: frozenset[str] = frozenset()) -> str:
    try:
        parsed = urlsplit(url)
        permitted_host = (parsed.hostname in HOSTS or
                          url in verified_cdn_urls and _EVENTHOSTS_PDF.fullmatch(url) is not None)
        valid = (parsed.scheme == 'https' and permitted_host
                 and parsed.username is None and parsed.password is None
                 and parsed.port in (None, 443) and not any(ord(c) < 33 for c in url))
    except ValueError:
        valid = False
    if not valid:
        raise PaperSummaryError('source_url_unsupported', 'source URL is outside supported official HTTPS hosts')
    return url


def _published(value: str) -> str:
    value = value.strip().replace('/', '-')
    if not re.fullmatch(r'\d{4}(?:-\d{2}(?:-\d{2})?)?', value):
        return ''
    try:
        parts = [int(part) for part in value.split('-')]
        date(*(parts + [1] * (3 - len(parts))))
    except ValueError:
        return ''
    return value


def _page_details(raw: bytes, url: str, title: str) -> dict:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, 'html.parser')
    meta = {str(tag.get('name', tag.get('property', ''))).casefold(): str(tag.get('content', ''))
            for tag in soup.select('meta[content]')}
    identities = [meta.get('citation_title', ''), meta.get('dc.title', '')]
    identities.extend(tag.get_text(' ', strip=True) for tag in soup.select('h1, h2, #papertitle'))
    expected = normalize_title(title)
    if not expected or not any(normalize_title(value) == expected for value in identities):
        raise PaperSummaryError('paper_identity_mismatch', 'official page title does not match the accepted paper')
    metadata = {}
    published = _published(meta.get('citation_publication_date', meta.get('dc.date.issued', '')))
    if published:
        metadata['published'] = published
    doi = meta.get('citation_doi', '').strip()
    if re.fullmatch(r'10\.\d{4,9}/\S+', doi):
        metadata['doi'] = doi
    abstract = meta.get('citation_abstract', '')
    if not abstract:
        node = soup.select_one('#abstract, .abstract, .abstract-content, .paper-abstract')
        if node:
            abstract = node.get_text(' ', strip=True)
    links = [meta.get('citation_pdf_url', '')]
    for anchor in soup.select('a[href]'):
        href = str(anchor['href'])
        label = anchor.get_text(' ', strip=True).casefold()
        if ('.pdf' in href.lower() or '/pdf' in href.lower()
                or label in {'pdf', 'paper', 'download pdf', 'openreview', 'arxiv'}
                or re.search(r'/article/(?:view|download)/', href)):
            links.append(href)
    permitted, verified_cdn_urls = [], []
    for link in links:
        if not link:
            continue
        absolute = urljoin(url, link)
        # These are explicit links on the identity-checked official page, not
        # title searches. Fetch the linked submission's full paper directly.
        arxiv = re.fullmatch(r'https://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v[1-9]\d*)?)', absolute)
        if arxiv:
            absolute = 'https://arxiv.org/pdf/' + arxiv.group(1)
        if absolute.startswith('https://openreview.net/forum?id='):
            absolute = absolute.replace('/forum?', '/pdf?', 1)
        try:
            verified_cdn = _official_cdn_link(absolute, url)
            _safe_url(absolute, verified_cdn_urls=frozenset({absolute}) if verified_cdn else frozenset())
        except PaperSummaryError:
            continue
        if absolute not in permitted:
            permitted.append(absolute)
            if verified_cdn:
                verified_cdn_urls.append(absolute)
    return {'metadata': metadata, 'links': permitted, 'abstract': abstract,
            'verified_cdn_urls': verified_cdn_urls}


def _download(url: str, directory: Path, *, verified_cdn_urls: frozenset[str] = frozenset()) -> tuple[bytes, str, str]:
    """Cache bounded raw responses, checking every redirect before requesting it."""
    _safe_url(url, verified_cdn_urls=verified_cdn_urls)
    key = hashlib.sha256(url.encode()).hexdigest()
    path, receipt = directory / (key + '.raw'), directory / (key + '.json')
    try:
        info = json.loads(receipt.read_text(encoding='utf-8'))
        raw = path.read_bytes() if path.stat().st_size <= MAX_BYTES else b''
        final = _safe_url(info['final_url'], verified_cdn_urls=verified_cdn_urls)
        if (info.get('url') == url and raw and info.get('sha256') == hashlib.sha256(raw).hexdigest()
                and (urlsplit(final).hostname == urlsplit(url).hostname
                     or (urlsplit(url).hostname, urlsplit(final).hostname) in REDIRECT_PAIRS)):
            return raw, str(info['content_type']), final
    except (OSError, ValueError, KeyError, TypeError):
        pass
    current = url
    with requests.Session() as session:
        session.trust_env = False
        session.proxies = requests.utils.get_environ_proxies(url)
        try:
            for _ in range(6):
                with session.get(current, allow_redirects=False, stream=True, timeout=(10, 60),
                                 headers={'User-Agent': 'LOKEN-conference-intake/1.0'}) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        target = _safe_url(urljoin(current, response.headers.get('Location', '')),
                                           verified_cdn_urls=verified_cdn_urls)
                        pair = (urlsplit(current).hostname, urlsplit(target).hostname)
                        if pair[0] != pair[1] and pair not in REDIRECT_PAIRS:
                            raise PaperSummaryError('source_redirect_rejected', 'official source redirected outside its permitted publisher')
                        current = target
                        continue
                    if response.status_code != 200:
                        raise PaperSummaryError('source_unavailable', 'official publisher source is currently inaccessible')
                    body = bytearray()
                    for chunk in response.iter_content(64 * 1024):
                        body.extend(chunk)
                        if len(body) > MAX_BYTES:
                            raise PaperSummaryError('source_too_large', 'official source exceeds the download boundary')
                    raw = bytes(body)
                    content_type = response.headers.get('Content-Type', '').split(';')[0].lower()
                    _atomic_write(path, raw)
                    _atomic_write(receipt, json.dumps({'url': url, 'final_url': current,
                        'sha256': hashlib.sha256(raw).hexdigest(), 'content_type': content_type},
                        ensure_ascii=False).encode())
                    return raw, content_type, current
        except requests.RequestException:
            raise PaperSummaryError('source_unavailable', 'official source request failed') from None
    raise PaperSummaryError('source_redirect_rejected', 'official source exceeded the redirect boundary')


def _pdf_document(raw: bytes, title: str, abstract: str = '') -> PaperDocument:
    document = extract_pdf_document(raw, title)
    # The shared extractor's overlap check is intentionally broad for arXiv.
    # Conference downloads require the exact normalized title in the first page.
    first = document.sections[0].text if document.sections else ''
    probe = re.split(r'\babstract\b', first, maxsplit=1, flags=re.I)[0]
    if normalize_title(title) not in normalize_title(probe[:5000]):
        raise PaperSummaryError('paper_identity_mismatch', 'PDF first page does not contain the accepted paper title')
    document = _without_running_headers(document)
    if not abstract:
        match = re.search(r'\babstract\b\s*[:.\-—]?\s*(.+?)(?:\n\s*(?:1[.\s]+)?introduction\b)',
                          first, re.I | re.S)
        abstract = match.group(1).strip() if match else ''
    abstract = re.sub(r'^abstract\s*[:.\-—]?\s*', '', abstract.strip(), flags=re.I)
    if not 40 <= len(abstract) <= 16000:
        raise PaperSummaryError('abstract_unavailable', 'official paper abstract is not usable')
    document = _validate_document(PaperDocument(document.title, abstract, document.sections), title)
    extract_introduction(document)
    return document


def _without_running_headers(document: PaperDocument) -> PaperDocument:
    """Remove evidenced PDF page furniture before strict section extraction.

    LNCS author headers such as ``4 L. Han et al.`` otherwise look like a
    numbered section. Only the first line of its exact numbered PDF page is
    eligible, and author headers must repeat on another page.
    """
    author_headers = []
    for index, section in enumerate(document.sections, 1):
        first = section.text.splitlines()[0] if section.text else ''
        match = re.fullmatch(rf'{index}\s+([A-Z]\.\s+[A-Za-z-]+\s+et al\.)', first)
        author_headers.append(match.group(1) if match and section.heading == f'Page {index}' else '')
    sections = []
    for index, section in enumerate(document.sections, 1):
        lines = section.text.splitlines()
        if lines and section.heading == f'Page {index}':
            author = author_headers[index - 1]
            title_header = re.fullmatch(rf'(.{{15,}})\s+{index}', lines[0])
            if (author and author_headers.count(author) >= 2 or title_header
                    and normalize_title(title_header.group(1)) in normalize_title(document.title)):
                lines = lines[1:]
        sections.append(PaperSection(section.heading, '\n'.join(lines)))
    return PaperDocument(document.title, document.abstract, tuple(sections))


def _crossref(url: str, title: str, directory: Path) -> tuple[dict, list[str]]:
    """Resolve only an explicit ACM/IEEE DOI, never a fuzzy bibliographic search."""
    match = re.search(r'(10\.(?:1145|1109)/[^?#]+)', unquote(url))
    if match is None:
        return {}, []
    doi = match.group(1)
    raw, _, _ = _download('https://api.crossref.org/works/' + quote(doi, safe=''), directory)
    try:
        item = json.loads(raw)['message']
        if (str(item.get('DOI', '')).casefold() != doi.casefold()
                or not any(normalize_title(t) == normalize_title(title) for t in item.get('title', []))):
            raise PaperSummaryError('paper_identity_mismatch', 'DOI metadata does not match the accepted paper')
        metadata = {'doi': doi}
        for field in ('published-online', 'published-print', 'published'):
            parts = item.get(field, {}).get('date-parts', [[]])[0]
            value = '-'.join(f'{int(part):04d}' if index == 0 else f'{int(part):02d}'
                             for index, part in enumerate(parts))
            if _published(value):
                metadata['published'] = value
                break
        links = [link['URL'] for link in item.get('link', [])
                 if link.get('content-type') == 'application/pdf' and isinstance(link.get('URL'), str)]
        return metadata, links
    except PaperSummaryError:
        raise
    except (ValueError, TypeError, KeyError, IndexError):
        raise PaperSummaryError('source_metadata_invalid', 'official DOI metadata could not be parsed') from None


def acquire_conference_paper(record: dict, cache_root: Path) -> tuple[AcquiredPaper, dict]:
    """Acquire full text; failure preserves accepted library membership upstream.

    ``cache_root`` is this record's directory under build/conferences/intake/sources.
    A record's calendar/edition date is never substituted for publication evidence.
    """
    identity, title = str(record.get('id', '')), str(record.get('title', '')).strip()
    if re.fullmatch(r'conf-[a-f0-9]{8,64}', identity) is None or not title:
        raise PaperSummaryError('source_record_invalid', 'conference source requires an internal ID and title')
    directory = Path(cache_root).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    url = _safe_url(str(record.get('url', '')))
    metadata, queue, abstract = {}, [url], ''
    arxiv_id = str(record.get('arxiv_id', ''))
    if arxiv_id:
        if re.fullmatch(r'\d{4}\.\d{4,5}(?:v[1-9]\d*)?', arxiv_id) is None:
            raise PaperSummaryError('invalid_arxiv_id', 'conference record contains an invalid arXiv ID')
        metadata['arxiv_id'] = arxiv_id.split('v')[0]
        queue.insert(0, 'https://arxiv.org/pdf/' + arxiv_id)
    if urlsplit(url).hostname in {'doi.org', 'dl.acm.org', 'ieeexplore.ieee.org'}:
        try:
            verified, links = _crossref(url, title, directory)
            metadata.update(verified)
            queue.extend(links)
        except PaperSummaryError as error:
            if error.code == 'paper_identity_mismatch':
                raise
    visited, verified_cdn_urls = set(), set()
    failure = PaperSummaryError('source_unavailable', 'no accessible official full paper source was found')
    for _ in range(10):
        if not queue:
            break
        target = queue.pop(0)
        if target in visited:
            continue
        visited.add(target)
        try:
            raw, content_type, final_url = _download(target, directory,
                verified_cdn_urls=frozenset(verified_cdn_urls))
            if raw.startswith(b'%PDF-'):
                document = _pdf_document(raw, title, abstract)
                arxiv = re.fullmatch(r'https://arxiv\.org/pdf/(\d{4}\.\d{4,5})(?:v[1-9]\d*)?(?:\.pdf)?', final_url)
                if arxiv:
                    metadata['arxiv_id'] = arxiv.group(1)
                path = directory / 'source.pdf'
                _atomic_write(path, raw)
                sha = hashlib.sha256(raw).hexdigest()
                _atomic_write(directory / 'verified.json', json.dumps({'id': identity, 'title': title,
                    'url': final_url, 'source_sha256': sha, 'metadata': metadata}, ensure_ascii=False).encode())
                return AcquiredPaper(identity, 'pdf', path, sha, document), metadata
            if content_type not in {'text/html', 'application/xhtml+xml'}:
                raise PaperSummaryError('source_unavailable', 'official response did not provide a paper document')
            details = _page_details(raw, final_url, title)
            verified_cdn_urls.update(details['verified_cdn_urls'])
            metadata.update(details['metadata'])
            abstract = details['abstract'] or abstract
            queue.extend(link for link in details['links'] if link not in visited)
        except PaperSummaryError as error:
            failure = error
    raise failure
