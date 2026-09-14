"""Resumable, private fixed-ID download/summary batches; never publishes or runs Git."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
import threading
from types import SimpleNamespace
from datetime import datetime, timezone
from uuid import uuid4

import requests

from . import workflow as batch
from papers.annotations.catalog import archive_paper_ids, load_annotation_catalog
from papers.paths import ANNOTATIONS
from papers.summaries.models import PaperSummaryError
from papers.summaries.paths import normalize_arxiv_id, private_path, run_lock
from papers.model_runtime import DEFAULT_MODEL, DEFAULT_MODEL_TIMEOUT_SECONDS, DEFAULT_MODEL_WORKERS, MAX_MODEL_WORKERS
from shared.loopback_chat import LoopbackChatError, validate_loopback_base_url
from shared.rendering import atomic_write_text


NON_RETRYABLE_RECOVERY_ERRORS = frozenset({
    'invalid_topic_review',
    'local_note_conflict',
})


@dataclass(frozen=True, slots=True)
class CycleStatus:
    phase: str
    processed: int
    total: int
    pending: int
    batch_ids: tuple[str, ...]
    history_count: int
    failed: bool
    network_paused: bool
    checkpoint: str

    def to_dict(self):
        return {
            'phase': self.phase,
            'processed': self.processed,
            'total': self.total,
            'pending': self.pending,
            'batch_ids': list(self.batch_ids),
            'history_count': self.history_count,
            'failed': self.failed,
            'network_paused': self.network_paused,
            'checkpoint': self.checkpoint,
        }


class DownloadGate:
    """Serialize request starts across workers, retries and PDF fallback."""
    def __init__(self, interval, *, clock=time.monotonic, sleeper=time.sleep):
        self.interval, self.clock, self.sleeper = interval, clock, sleeper
        self.next_request = 0.0
        self.blocked = False
        self._request_lock = threading.Lock()

    def before_request(self):
        with self._request_lock:
            self._wait_for_slot()

    def _wait_for_slot(self):
        if self.blocked:
            raise PaperSummaryError('source_throttled', 'network paused after server refused requests')
        remaining = self.next_request - self.clock()
        if remaining > 0:
            self.sleeper(remaining)
        self.next_request = self.clock() + self.interval

    def after_response(self, status):
        if status in {403, 429, 503}:
            self.blocked = True
            raise PaperSummaryError('source_throttled', f'arXiv HTTP {status}; stop and retry later')

    def session(self):
        return PacedSession(self)


class PacedSession(requests.Session):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def request(self, method, url, **kwargs):
        self.gate.before_request()
        response = super().request(method, url, **kwargs)
        try:
            self.gate.after_response(response.status_code)
        except PaperSummaryError:
            response.close()
            raise
        return response


def stage_args(args, mode, paper_ids, gate=None):
    return SimpleNamespace(mode=mode, paper=list(paper_ids), limit=None, dry_run=False,
        workers=1 if mode == 'download' else args.workers, model=args.model,
        base_url=args.base_url, timeout=args.timeout,
        download_session_factory=gate.session if gate else None)


def cached_ids(ids):
    """Only valid shared source caches from this exact batch may reach inference."""
    ready = set()
    for item in batch.archive_candidates(batch.DEFAULT_ARCHIVE, batch.DEFAULT_LEDGER, tuple(ids)):
        if item.arxiv_id not in ready and batch.ArxivSourceClient()._load_cached(item.arxiv_id, item.title):
            ready.add(item.arxiv_id)
    return [paper_id for paper_id in ids if paper_id in ready]


def cycle_state_path():
    return private_path('batch', 'cycle-state.json')


def save_state(state):
    atomic_write_text(cycle_state_path(),
                      json.dumps(state, ensure_ascii=False, indent=2) + '\n')


def prioritized_queue(selected, priority_topic='Relighting'):
    """Deduplicate a date-sorted selection while putting one topic first."""
    items = [item for item, _ in selected]
    items.sort(key=lambda item: (-item.updated.toordinal(), item.arxiv_id))
    priority_ids = {item.arxiv_id for item in items if item.topic == priority_topic}
    priority = [item.arxiv_id for item in items if item.arxiv_id in priority_ids]
    remainder = [item.arxiv_id for item in items if item.arxiv_id not in priority_ids]
    return list(dict.fromkeys(priority + remainder))


def retryable_failure_ids(history):
    """Return prior-cycle failed papers that are safe to retry automatically."""
    result = set()
    for batch_record in history:
        for field in ('download_report', 'summary_report'):
            raw_path = batch_record.get(field)
            if not raw_path:
                continue
            try:
                path = Path(raw_path)
                if path.stat().st_size > 2 * 1024 * 1024:
                    raise ValueError('oversize report')
                report = json.loads(path.read_text(encoding='utf-8'))
                records = report['records']
                if not isinstance(records, list):
                    raise ValueError('invalid records')
                for record in records:
                    if (not isinstance(record, dict) or 'id' not in record
                            or record.get('status') not in {'succeeded', 'failed'}):
                        raise ValueError('invalid record')
                    if (record['status'] == 'failed'
                            and record.get('error') not in NON_RETRYABLE_RECOVERY_ERRORS):
                        result.add(normalize_arxiv_id(record['id']))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                raise PaperSummaryError(
                    'invalid_cycle_history',
                    'completed batch reports are missing or invalid; preserve the checkpoint and inspect',
                ) from None
    return result


def current_archive_ids():
    try:
        archive = json.loads(batch.DEFAULT_ARCHIVE.read_text(encoding='utf-8'))
        if not isinstance(archive, dict):
            raise ValueError('invalid archive')
        return archive_paper_ids(archive)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        raise PaperSummaryError(
            'invalid_recovery_inputs',
            'archive cannot be read safely for recovery',
        ) from None


def missing_annotation_ids(*, archived_ids=None):
    """Return archived papers with no public annotation record."""
    archived = current_archive_ids() if archived_ids is None else set(archived_ids)
    try:
        annotations = load_annotation_catalog(ANNOTATIONS, batch.PAPER_LABELS)
        return archived - set(annotations)
    except (OSError, ValueError, TypeError):
        raise PaperSummaryError(
            'invalid_recovery_inputs',
            'public annotation catalog cannot be read safely',
        ) from None


def existing_recovery_targets(*, retryable_ids, missing_annotation_ids, archived_ids):
    """Never resurrect a paper removed from every current archive topic."""
    return (set(retryable_ids) | set(missing_annotation_ids)) & set(archived_ids)


def ordered_recovery_queue(selected, *, retryable_ids, missing_annotation_ids):
    """Retry failures first, then unlabelled papers; prioritize Relighting in each group."""
    retryable = [(item, state) for item, state in selected if item.arxiv_id in retryable_ids]
    retry_queue = prioritized_queue(retryable)
    retry_set = set(retry_queue)
    missing = [(item, state) for item, state in selected
               if item.arxiv_id in missing_annotation_ids and item.arxiv_id not in retry_set]
    return retry_queue + prioritized_queue(missing)


def needs_recovery_cycle(state):
    """A regular completed checkpoint is followed by exactly one recovery cycle."""
    return bool(state and state.get('phase') == 'complete'
                and state.get('cycle_kind') != 'recovery')


def reordered_cycle_state(old, queue):
    """Reset queue progress without turning an active recovery into a regular cycle."""
    state = {
        'version': 1,
        'queue': list(queue),
        'offset': 0,
        'batch_ids': [],
        'phase': 'download' if queue else 'complete',
        'history': old.get('history', []) if old else [],
    }
    if old:
        for field in ('cycle_kind', 'recovery', 'previous_checkpoint_backup'):
            if field in old:
                state[field] = old[field]
    return state


def reorder_requested_ids(old):
    """A recovery policy refresh must reconsider every original recovery target."""
    if old and old.get('cycle_kind') == 'recovery':
        return list(old['queue'])
    return []


def recovery_cycle_pending():
    return needs_recovery_cycle(_read_state(include_complete=True))


def _recovery_state(args, completed, *, apply):
    raw_retryable = retryable_failure_ids(completed.get('history', []))
    archived = current_archive_ids()
    missing = missing_annotation_ids(archived_ids=archived)
    targets = existing_recovery_targets(
        retryable_ids=raw_retryable,
        missing_annotation_ids=missing,
        archived_ids=archived,
    )
    retryable = raw_retryable & targets
    selected = []
    skipped = 0
    if targets:
        selected, skipped = batch.select_items(stage_args(args, 'summarize', sorted(targets)))
    queue = ordered_recovery_queue(
        selected,
        retryable_ids=retryable,
        missing_annotation_ids=missing,
    )
    state = {
        'version': 1,
        'cycle_kind': 'recovery',
        'queue': queue,
        'offset': 0,
        'batch_ids': [],
        'phase': 'download' if queue else 'complete',
        'history': [],
        'recovery': {
            'retryable_failures': len(retryable),
            'removed_failures_excluded': len(raw_retryable - archived),
            'missing_annotations': len(missing),
            'selected': len(queue),
            'completed_skipped': skipped,
        },
    }
    if apply:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup = batch.ROOT / 'build' / 'reports' / f'cycle-state-before-recovery-{stamp}-{uuid4().hex[:8]}.json'
        atomic_write_text(backup, json.dumps(completed, ensure_ascii=False, indent=2) + '\n')
        state['previous_checkpoint_backup'] = str(backup)
        save_state(state)
        print(json.dumps({'event': 'recovery_cycle_started', **state['recovery'],
                          'checkpoint_backup': str(backup)}, ensure_ascii=False, sort_keys=True), flush=True)
    return state


def _initial_state(args, *, apply):
    completed = _read_state(include_complete=True)
    if needs_recovery_cycle(completed):
        return _recovery_state(args, completed, apply=apply)
    if completed is not None:
        return None
    selected, _ = batch.select_items(stage_args(args, 'summarize', []))
    state = {'version': 1, 'queue': prioritized_queue(selected), 'offset': 0,
             'batch_ids': [], 'phase': 'download', 'history': []}
    if apply:
        save_state(state)
    return state


def reorder_checkpoint(args, *, apply):
    """Preview or atomically replace the active queue from current durable results."""
    old = _read_state(include_complete=True)
    selected, skipped = batch.select_items(
        stage_args(args, 'summarize', reorder_requested_ids(old)),
    )
    queue = prioritized_queue(selected)
    relighting = {item.arxiv_id for item, _ in selected if item.topic == 'Relighting'}
    summary = {
        'action': 'apply' if apply else 'dry-run',
        'old_total': len(old['queue']) if old else 0,
        'old_processed': old['offset'] if old else 0,
        'new_pending': len(queue),
        'completed_skipped': skipped,
        'relighting_pending': len(relighting),
        'next_papers': queue[:20],
    }
    if apply:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup = batch.ROOT / 'build' / 'reports' / f'cycle-state-before-reorder-{stamp}-{uuid4().hex[:8]}.json'
        if old is not None:
            atomic_write_text(backup, json.dumps(old, ensure_ascii=False, indent=2) + '\n')
            summary['backup'] = str(backup)
        state = reordered_cycle_state(old, queue)
        save_state(state)
        summary['checkpoint'] = str(cycle_state_path())
    return summary


def _read_state(*, include_complete):
    path = cycle_state_path()
    if not path.exists():
        return None
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError('oversize state')
        state = json.loads(path.read_text(encoding='utf-8'))
        queue, offset, ids = state['queue'], state['offset'], state['batch_ids']
        if (state['version'] != 1 or state['phase'] not in {'download', 'summarize', 'complete'}
                or not isinstance(queue, list) or len(queue) != len(set(queue))
                or any(normalize_arxiv_id(i) != i for i in queue)
                or type(offset) is not int or not 0 <= offset <= len(queue)
                or not isinstance(ids, list) or len(ids) > 100
                or ids != queue[offset:offset + len(ids)]
                or not isinstance(state['history'], list)):
            raise ValueError('invalid state')
        if state['phase'] == 'complete':
            if offset != len(queue) or ids:
                raise ValueError('invalid completion')
            return state if include_complete else None
        return state
    except (ValueError, TypeError, KeyError, OSError, PaperSummaryError):
        raise PaperSummaryError('invalid_cycle_state', 'cycle checkpoint is invalid; preserve it and inspect before retry') from None


def load_state():
    """Load an active v1 checkpoint; completed checkpoints start a fresh queue."""
    return _read_state(include_complete=False)


def get_cycle_status():
    """Return machine-readable checkpoint status without selecting work or writing."""
    state = _read_state(include_complete=True)
    path = cycle_state_path()
    if state is None:
        return CycleStatus('idle', 0, 0, 0, (), 0, False, False, str(path))
    history = state['history']
    return CycleStatus(
        state['phase'], state['offset'], len(state['queue']),
        len(state['queue']) - state['offset'], tuple(state['batch_ids']), len(history),
        bool(state.get('download_failed')) or any(bool(item.get('failed')) for item in history),
        bool(state.get('network_paused')), str(path),
    )


def cycle_status():
    return get_cycle_status()


def _run_cycle(args, *, max_batches):
    state = load_state()
    if args.dry_run:
        if state is None:
            preview = _initial_state(args, apply=False)
            queue = preview['queue'] if preview else []
        else:
            queue = state['queue'][state['offset']:]
        print(f'pending_unique={len(queue)} batch_size={args.batch_size} '
              f'download_workers=1 request_interval={args.download_interval}s summary_workers={args.workers}')
        print('next_batch=' + ','.join(queue[:args.batch_size]))
        return 0
    if state is None:
        state = _initial_state(args, apply=True)
        if state is None:
            print(f'cycle=complete processed=0/0 checkpoint={cycle_state_path()}', flush=True)
            return 0
    gate = DownloadGate(args.download_interval)
    completed = 0
    while state['offset'] < len(state['queue']):
        if not state['batch_ids']:
            state['batch_ids'] = state['queue'][state['offset']:state['offset'] + args.batch_size]
            state['phase'] = 'download'
            save_state(state)
        ids = state['batch_ids']
        print(f"Batch {len(state['history'])+1}: {len(ids)} papers; "
              f"progress={state['offset']}/{len(state['queue'])}; phase={state['phase']}", flush=True)
        if state['phase'] == 'download':
            report = {}
            batch.run(stage_args(args, 'download', ids, gate), report_sink=report)
            records = report.get('records', [])
            blocked = any(r.get('error') == 'source_throttled' for r in records)
            attempted = [r for r in records if r.get('error') != 'local_note_conflict']
            unavailable = bool(attempted) and all(r.get('error') == 'source_unavailable' for r in attempted)
            state.update(phase='summarize', download_report=report.get('report_path'),
                         download_failed=any(r['status'] == 'failed' for r in records),
                         network_paused=blocked or unavailable)
            save_state(state)
        ready = cached_ids(ids)
        report = {}
        if ready:
            batch.run(stage_args(args, 'summarize', ready), report_sink=report)
        records = report.get('records', [])
        if any(r.get('error') in {'model_unavailable', 'model_http_error'} for r in records):
            print('Local model unavailable or returned an HTTP error. Check service/model settings and rerun; '
                  'this batch resumes before more downloads.', flush=True)
            return 3
        if state.get('network_paused'):
            state['phase'] = 'download'
            save_state(state)
            print('Network paused after refusal or batch-wide failure. Cached papers were summarized; '
                  'retry later after the service/network recovers. No further download batch was started.', flush=True)
            return 3
        state['history'].append({'ids': ids, 'download_report': state.get('download_report'),
            'summary_report': report.get('report_path'),
            'failed': state.get('download_failed', False) or any(r['status'] == 'failed' for r in records)})
        state['offset'] += len(ids)
        state['batch_ids'] = []
        state['phase'] = 'download' if state['offset'] < len(state['queue']) else 'complete'
        save_state(state)
        completed += 1
        if max_batches is not None and completed >= max_batches:
            break
        if state['phase'] != 'complete' and args.batch_pause:
            print(f'Batch finished; pausing {args.batch_pause:g}s before the next download batch.', flush=True)
            time.sleep(args.batch_pause)
    if state['offset'] == len(state['queue']):
        state['phase'] = 'complete'
        save_state(state)
    print(f"cycle={state['phase']} processed={state['offset']}/{len(state['queue'])} "
          f"checkpoint={cycle_state_path()}", flush=True)
    return 3 if any(h['failed'] for h in state['history']) else 0


def run_cycle(args):
    return _run_cycle(args, max_batches=args.max_batches)


def run_one_batch(args):
    """Run or resume exactly one fixed-ID batch, returning at its priority boundary."""
    return _run_cycle(args, max_batches=1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=100, help='unique papers per batch, 1-100')
    parser.add_argument('--download-interval', type=float, default=5, help='minimum seconds between requests, at least 3')
    parser.add_argument('--batch-pause', type=float, default=60, help='pause after summary and before next download batch')
    parser.add_argument('--workers', type=int, default=DEFAULT_MODEL_WORKERS,
                        help=f'summary workers, 1-{MAX_MODEL_WORKERS}; downloads are always serial')
    parser.add_argument('--max-batches', type=int, help='stop after N batches, keep checkpoint; default runs entire queue')
    parser.add_argument('--model', default=os.environ.get('TOGOS_WSL_LLM_MODEL', DEFAULT_MODEL))
    parser.add_argument('--base-url', default=os.environ.get('TOGOS_WSL_LLM_BASE_URL', 'http://127.0.0.1:8000/v1'))
    parser.add_argument('--timeout', type=float, default=DEFAULT_MODEL_TIMEOUT_SECONDS)
    parser.add_argument('--reorder-checkpoint', choices=('dry-run', 'apply'),
                        help='rebuild pending queue with Relighting first, then global newest-first')
    parser.add_argument('--dry-run', action='store_true', help='show next batch without download, inference or writes')
    parser.add_argument('--status', action='store_true', help='print checkpoint status as JSON; no writes')
    args = parser.parse_args(argv)
    if args.status:
        return args
    if (not 1 <= args.batch_size <= 100 or not 1 <= args.workers <= MAX_MODEL_WORKERS
            or not math.isfinite(args.download_interval) or args.download_interval < 3
            or not math.isfinite(args.batch_pause) or args.batch_pause < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or args.max_batches is not None and args.max_batches < 1 or not args.model.strip()):
        parser.error('invalid batch size, workers, interval, pause, timeout or model')
    validate_loopback_base_url(args.base_url)
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
        if args.status:
            print(json.dumps(get_cycle_status().to_dict(), ensure_ascii=False, sort_keys=True))
            return 0
        if args.reorder_checkpoint:
            with run_lock():
                print(json.dumps(reorder_checkpoint(args, apply=args.reorder_checkpoint == 'apply'),
                                 ensure_ascii=False, sort_keys=True))
            return 0
        if args.dry_run:
            return run_cycle(args)
        with run_lock():
            return run_cycle(args)
    except (PaperSummaryError, LoopbackChatError) as error:
        print(f'error {error.code}: {error.message}', flush=True)
        return 2
    except OSError:
        print('Local I/O failed; preserve checkpoint and inspect disk/path permissions.', flush=True)
        return 2
    except KeyboardInterrupt:
        print('Interrupted. Checkpoint and successful caches retained; rerun the same command to resume.', flush=True)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
