"""Public conference overlay without synthetic arXiv identities."""
from __future__ import annotations

from datetime import date
import html
import json
from pathlib import Path
import re

from papers.paths import ROOT
from papers.proceedings import normalize_title
from shared.rendering import atomic_write_text
from shared.site_shell import render_site_page

LIBRARY = ROOT / 'content/papers/conference-library.json'


def conference_annotation(record):
    from papers.annotations.catalog import (load_annotation_definitions, load_topic_tag_allowlists,
        annotation_from_value, filter_annotation_for_topics)
    if not record.get('annotation'):
        return None
    labels = load_annotation_definitions(ROOT / 'config/site.yaml')
    allowlists = load_topic_tag_allowlists(ROOT / 'config/site.yaml', labels)
    value = annotation_from_value(record['id'], record['annotation'], labels)
    if not record['topics']:
        # Excluded records retain historical annotations for audit; library_rows
        # omits them, so no active-topic tag allowlist is applicable.
        return value
    return filter_annotation_for_topics(value, labels, allowlists, record['topics'])


def load_library(path: Path = LIBRARY) -> dict:
    if not path.exists():
        return {'version': 1, 'papers': {}}
    value = json.loads(path.read_text(encoding='utf-8'))
    return validate_library(value)


def validate_library(value: dict) -> dict:
    if value.get('version') != 1 or not isinstance(value.get('papers'), dict):
        raise ValueError('Invalid conference library')
    for key, record in value['papers'].items():
        review = record.get('topic_review', {}).get('decisions', {})
        excluded = bool(review) and all(accept is False for accept in review.values())
        if (not re.fullmatch(r'conf-[a-f0-9]{24}', key) or record.get('id') != key
                or not record.get('title') or (not record.get('topics') and not excluded)
                or not record.get('url', '').startswith('https://')):
            raise ValueError('Invalid conference library record')
        display_id(record)
        if record.get('annotation'):
            conference_annotation(record)
    return value


def display_id(record: dict) -> str:
    arxiv_id = record.get('arxiv_id')
    if arxiv_id:
        if not re.fullmatch(r'\d{4}\.\d{4,5}', arxiv_id):
            raise ValueError('Invalid real arXiv ID')
        return arxiv_id
    parts = record['published'].split('-')
    if not re.fullmatch(r'\d{4}(?:-\d{2}){0,2}', record['published']):
        raise ValueError('Invalid partial publication date')
    date(int(parts[0]), int(parts[1]) if len(parts) > 1 else 1,
         int(parts[2]) if len(parts) > 2 else 1)
    return f"{parts[0][-2:]}{parts[1] if len(parts) > 1 else 'XX'}.{parts[2] if len(parts) > 2 else 'XX'}XXX"


def existing_identities(archive: dict, ledger: dict) -> tuple[set, set]:
    from papers.site import parse_entry
    ids, titles = set(), set()
    for entries in archive.values():
        for key, entry in entries.items():
            ids.add(key)
            titles.add(normalize_title(parse_entry(key, entry)['title']))
    for key, entry in ledger.get('papers', {}).items():
        if entry.get('status') == 'rejected':
            ids.add(key)
            titles.add(normalize_title(entry.get('title', '')))
    return ids, titles


def library_rows(library: dict, archive: dict, ledger: dict) -> list[dict]:
    ids, titles = existing_identities(archive, ledger)
    rows = []
    for record in library['papers'].values():
        if not record['topics']:
            continue
        title = normalize_title(record['title'])
        if record.get('arxiv_id') in ids or title in titles:
            continue
        parts = [int(part) for part in record['published'].split('-')]
        annotation = conference_annotation(record)
        rows.append({
            'id': record['id'], 'display_id': display_id(record), 'title': record['title'],
            'date': date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1),
            'date_precision': len(parts), 'published_at': record['published'], 'conference_library': True,
            'date_source': record.get('date_source', {}),
            'paper_url': 'https://arxiv.org/abs/' + record['arxiv_id'] if record.get('arxiv_id') else record['url'],
            'authors': record.get('authors', ''), 'code_url': None,
            'topics': tuple(record['topics']), 'tags': annotation.tags if annotation else (),
            'institutions': annotation.institutions if annotation else (),
            'paper_type': annotation.paper_type if annotation else 'paper',
            'annotation_status': 'ready' if annotation and annotation.tags else 'pending', 'summary_pending': True,
            'conferences': record['conferences'],
        })
        if record.get('arxiv_id'):
            ids.add(record['arxiv_id'])
        titles.add(title)
    return rows


def retain_library_matches(rows: list[dict], library: dict, ledger: dict) -> None:
    """Keep an already-visible archive duplicate browsable after metadata enrichment."""
    rejected_ids, rejected_titles = existing_identities({}, ledger)
    by_id, by_title = {}, {}
    for record in library['papers'].values():
        title = normalize_title(record['title'])
        if record.get('arxiv_id') in rejected_ids or title in rejected_titles:
            continue
        if record.get('arxiv_id'):
            by_id[record['arxiv_id']] = record
        by_title[title] = record
    for row in rows:
        record = by_id.get(row['id']) or by_title.get(normalize_title(row['title']))
        if record:
            row['conference_library'] = True
            row['conferences'] = list({(c['edition'], c['url']): c for c in
                                       row.get('conferences', []) + record['conferences']}.values())


def publish_library_notes(library: dict, output_root: Path, visible_ids: set[str]) -> dict:
    """Use the existing summary panel contract and keep pending entries explicit."""
    from papers.summaries.models import PaperSummary
    papers, articles = {}, []
    for key, record in library['papers'].items():
        if key not in visible_ids:
            continue
        summary = record.get('summary')
        papers[key] = {'status': 'pending'}
        if not summary:
            continue
        validated = PaperSummary(summary['one_sentence'], summary['problem'], tuple(summary['contributions']))
        url = f'notes/conference-library.html#summary-{key}'
        papers[key] = {'status': 'ready', 'url': url}
        contributions = ''.join(f'<li>{html.escape(x)}</li>' for x in validated.contributions)
        articles.append(f'<article id="summary-{key}" class="paper-summary"><h2>{html.escape(record["title"])}</h2>'
                        f'<p>{html.escape(validated.one_sentence)}</p><h3>解决的问题</h3><p>{html.escape(validated.problem)}</p>'
                        f'<h3>主要贡献</h3><ul>{contributions}</ul></article>')
    manifest = json.dumps({'topic': 'conference', 'papers': papers}, ensure_ascii=False).replace('<', '\\u003c')
    destination = output_root / 'notes/conference-library.html'
    document = render_site_page(output_file=destination, output_root=output_root, active_section='learning',
                               page_title='会议论文要点', meta_description='会议论文的正文要点总结',
                               main_content='\n'.join(articles) + f'<script type="application/json" id="summary-catalog">{manifest}</script>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, document)
    return papers
