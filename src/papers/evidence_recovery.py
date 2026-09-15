"""Targeted, resumable recovery from verified official abstracts."""
from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from papers import paths, recheck as r, runtime
from papers.annotations.catalog import annotation_value, annotation_from_value
from papers.conference_library import LIBRARY, validate_library
from papers.conference_recheck import apply_decision
from papers.conference_sources import _download, _page_details, _pdf_document, acquire_conference_paper
from papers.summaries.acquisition import ArxivSourceClient
from papers.summaries.models import PaperSummaryError
from papers.site import parse_entry

SOURCE_LOCK = threading.Lock()


def official_abstract(item, directory):
    directory.mkdir(parents=True, exist_ok=True)
    if item['source'] == 'arXiv':
        client = ArxivSourceClient()
        try:
            try:
                abstract = client.acquire_abstract(item['id'], item['title'])
            except PaperSummaryError as error:
                if error.code == 'paper_identity_mismatch':
                    raise
                abstract = client.acquire(item['id'], item['title']).document.abstract
            return {'id': item['id'], 'title': item['title'], 'abstract': abstract,
                    'source_url': item['url'], 'basis': 'identity_verified_arxiv'}
        finally:
            client.session.close()
    record = item['record']
    try:
        raw, kind, final = _download(record['url'], directory)
        if 'html' in kind:
            abstract = _page_details(raw, final, record['title'])['abstract']
        elif raw.startswith(b'%PDF-'):
            abstract = _pdf_document(raw, record['title']).abstract
        else:
            abstract = ''
        if len(abstract.strip()) >= 40:
            return {'id': item['id'], 'title': item['title'], 'abstract': abstract,
                    'source_url': final, 'basis': 'identity_verified_publisher'}
    except PaperSummaryError:
        pass
    source, _ = acquire_conference_paper(record, directory)
    return {'id': item['id'], 'title': item['title'], 'abstract': source.document.abstract,
            'source_url': record['url'], 'basis': 'identity_verified_full_text'}


def work(item, directory, labels, allowlists, args):
    receipt = directory / (item['id'] + '.json')
    fingerprint = r.digest({'item': item, 'config': paths.CONFIG.read_text(encoding='utf-8'), 'model': args.model})
    if receipt.exists():
        old = r.read(receipt)
        if old.get('fingerprint') == fingerprint and 'status' in old:
            return old
    result = {'id': item['id'], 'source': item['source'], 'fingerprint': fingerprint, 'item': item}
    try:
        with SOURCE_LOCK:
            time.sleep(3)
            material = official_abstract(item, directory / 'sources' / item['id'])
        if not 40 <= len(material['abstract'].strip()) <= 16000:
            raise PaperSummaryError('abstract_unavailable', 'no bounded verified official abstract')
        result['material'] = material
        # Persist the extracted evidence before inference, including failed model attempts.
        r.atomic_write_json(receipt, result)
        topics = list(allowlists) if item['review'] else item['topics']
        allowed = r.annotation_labels_for_topics(labels, allowlists, topics)
        if item['review']:
            material['requested_topics'] = topics
            system = r.annotation_messages(item['title'], material['abstract'], allowed)[0]['content']
            system += '\n逐一复核所有主题。排除主要医学、临床、生物医学应用；证据不足用null。依据主要方法、综述对象或评估任务。输出decisions和annotation两个字段。每个reason至少40字符。'
            def save(attempts):
                result['attempts'] = attempts
                r.atomic_write_json(receipt, result)
            decisions, annotation, _ = r.infer_review(system, material, allowed, args, save)
            result.update(decisions=decisions, annotation=annotation_value(annotation))
            accepted = [t for t,v in decisions.items() if v['accept'] is True]
            result['status'] = 'uncertain' if any(v['accept'] is None for v in decisions.values()) else 'ready'
            if accepted and not annotation.tags:
                result['status'] = 'annotation_tags_missing'
        else:
            tags = r.evidence_tags(material, allowed, args)
            value = {'topics': item['topics'], 'tags': list(tags), 'paper_type': item.get('paper_type','paper'), 'institutions': []}
            result.update(annotation=annotation_value(annotation_from_value(item['id'], value, labels)), status='ready')
    except Exception as error:
        result.update(status=getattr(error, 'code', type(error).__name__), error=str(error))
    r.atomic_write_json(receipt, result)
    print(json.dumps({'id': item['id'], 'status': result['status']}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-list', type=Path, required=True)
    parser.add_argument('--month', default='2026-09')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    r.RUN_NAME = 'evidence-recovery-20260916'
    directory = paths.ROOT / 'build/paper-summaries/evidence-recovery-20260916'
    directory.mkdir(parents=True, exist_ok=True)
    model_args = SimpleNamespace(model=r.DEFAULT_MODEL, base_url='http://127.0.0.1:8000/v1', timeout=900)
    with runtime.lock('recheck-owner.lock'), runtime.lock('runtime.lock'):
        with r.run_lock():
            original = {k:r.read(p) for k,p in r.targets().items()}
            original['library'] = r.read(LIBRARY)
            if args.apply:
                r.atomic_write_json(runtime.PRIVATE / 'recheck/local-only.json', {'reason':'Targeted abstract recovery in progress'})
        labels, allowlists, aliases = r.taxonomy()
        items = {}
        for entry in r.read(args.review_list)['papers']:
            item = {**entry, 'review':True}
            if entry['source'] != 'arXiv':
                item['record'] = original['library']['papers'][entry['id']]
            items[item['id']] = item
        for topic, rows in original['archive'].items():
            for key,row in rows.items():
                if parse_entry(key,row)['date'].isoformat().startswith(args.month) and not original['annotations']['papers'].get(key,{}).get('tags'):
                    if key in items: continue
                    topics = list(dict.fromkeys(aliases[t] for t,rr in original['archive'].items() if key in rr))
                    items[key] = {'id':key,'title':parse_entry(key,row)['title'],'url':'https://arxiv.org/abs/'+key,'source':'arXiv','review':False,'topics':topics}
        for key,record in original['library']['papers'].items():
            if record['topics'] and record['published'].startswith(args.month) and not record.get('annotation',{}).get('tags') and key not in items:
                items[key] = {'id':key,'title':record['title'],'url':record['url'],'source':'会议','review':False,'topics':record['topics'],'record':record}
        r.atomic_write_json(directory/'scope.json', list(items.values()))
        with runtime.model_service('vllm-paper.service') as model:
            if model != model_args.model: raise ValueError('wrong registered model')
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda item:work(item,directory,labels,allowlists,model_args),items.values()))
        r.atomic_write_json(directory/'results.json', results)
        if not args.apply: return
        after = copy.deepcopy(original)
        for result in results:
            if 'annotation' not in result: continue
            item=result['item']; key=item['id']
            if item['source']=='arXiv':
                if item['review']:
                    after['archive'],after['ledger'],after['annotations']['papers']=r.apply_records(after['archive'],after['ledger'],after['annotations']['papers'],[result])
                else:
                    previous=after['annotations']['papers'].get(key,{})
                    value=result['annotation']; value['institutions']=previous.get('institutions',[])
                    after['annotations']['papers'][key]=value
            else:
                record=after['library']['papers'][key]
                if item['review']:
                    decisions={t:v['accept'] for t,v in result['decisions'].items()}
                    apply_decision(record,decisions,model_args.model)
                    record['topic_review']['reason']='；'.join(t+'：'+v['reason'] for t,v in result['decisions'].items())
                if record['topics']:
                    value=result['annotation']; value['topics']=record['topics']
                    value['institutions']=record.get('annotation',{}).get('institutions',[])
                    record['annotation']=value
        validate_library(after['library'])
        targets={**r.targets(),'library':LIBRARY}
        with r.run_lock():
            for key,path in targets.items():
                if r.read(path)!=original[key]: raise ValueError('concurrent public change; results preserved')
            # Preserve both complete snapshots before the first public write.
            r.atomic_write_json(directory/'publication.json', {'before':original,'after':after})
            for key,path in targets.items(): r.atomic_write_json(path,after[key])
        from papers.__main__ import build
        build()
        r.atomic_write_json(directory/'applied.json', {'count':len(results),'pending':sum(x['status']!='ready' for x in results)})


if __name__ == '__main__': main()
