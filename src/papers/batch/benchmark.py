"""Benchmark loopback-model continuous batching using cached Relighting papers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

from papers.model_runtime import DEFAULT_MODEL, DEFAULT_MODEL_TIMEOUT_SECONDS, MAX_MODEL_WORKERS
from papers.summaries.acquisition import ArxivSourceClient
from papers.summaries.models import PaperSummaryError
from papers.summaries.prompts import build_chunks, map_messages
from papers.summaries.summarizer import _complete
from shared.loopback_chat import LoopbackChatError, LoopbackChatTransport, validate_loopback_base_url
from shared.rendering import atomic_write_text
from .workflow import DEFAULT_ARCHIVE, DEFAULT_LEDGER, ROOT
from .catalog import archive_candidates

DEFAULT_LEVELS = (2, 4, 8, 12, 16)


def samples_for_level(samples, workers):
    """Use one saturated wave so each level has a bounded wall-clock cost."""
    return samples[:workers]


def recommend_workers(results, *, max_failure_rate=0.02, minimum_gain=0.05):
    stable = sorted((row for row in results if row['failure_rate'] <= max_failure_rate),
                    key=lambda row: row['workers'])
    if not stable:
        raise PaperSummaryError('benchmark_unstable', 'no concurrency level met the failure-rate limit')
    choice = stable[0]
    for row in stable[1:]:
        gain = row['papers_per_hour'] / choice['papers_per_hour'] - 1 if choice['papers_per_hour'] else math.inf
        if gain < minimum_gain:
            break
        choice = row
    return choice['workers']


def _percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _gpu_sample():
    try:
        raw = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            check=True, text=True, stdout=subprocess.PIPE, timeout=5,
        ).stdout.splitlines()
        values = [tuple(int(part.strip()) for part in line.split(',')) for line in raw if line.strip()]
        return max((item[0] for item in values), default=0), max((item[1] for item in values), default=0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None


def _cached_samples(limit):
    candidates = [item for item in archive_candidates(DEFAULT_ARCHIVE, DEFAULT_LEDGER)
                  if item.topic == 'Relighting']
    available = []
    client = ArxivSourceClient()
    try:
        seen = set()
        for item in candidates:
            if item.arxiv_id in seen:
                continue
            seen.add(item.arxiv_id)
            source = client._load_cached(item.arxiv_id, item.title)
            if source is not None:
                try:
                    chunks = build_chunks(source.document)
                except PaperSummaryError:
                    continue
                available.append((len(chunks), item.arxiv_id, source.document.title,
                                  chunks[len(chunks) // 2]))
    finally:
        client.session.close()
    if len(available) < limit:
        raise PaperSummaryError('benchmark_samples_missing',
                                f'need {limit} cached Relighting papers; found {len(available)}')
    available.sort()
    if limit == 1:
        return [available[len(available) // 2]]
    indexes = [round(i * (len(available) - 1) / (limit - 1)) for i in range(limit)]
    return [available[index] for index in indexes]


def _run_level(samples, workers, model, base_url, timeout):
    stop = threading.Event()
    gpu = []

    def monitor():
        while not stop.wait(0.5):
            gpu.append(_gpu_sample())

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    latencies, errors = [], []
    started = time.monotonic()

    def infer(sample):
        _, paper_id, title, chunk = sample
        begin = time.monotonic()
        try:
            transport = LoopbackChatTransport(base_url)
            _complete(transport, map_messages(title, chunk), model=model, timeout=timeout)
            return paper_id, time.monotonic() - begin, None
        except (PaperSummaryError, LoopbackChatError) as error:
            return paper_id, time.monotonic() - begin, error.code

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for future in as_completed(executor.submit(infer, sample) for sample in samples):
                _, latency, error = future.result()
                latencies.append(latency)
                if error:
                    errors.append(error)
    finally:
        stop.set()
        watcher.join(timeout=2)
    elapsed = time.monotonic() - started
    succeeded = len(samples) - len(errors)
    memory = [item[0] for item in gpu if item[0] is not None]
    utilization = [item[1] for item in gpu if item[1] is not None]
    return {
        'workers': workers, 'sample_size': len(samples), 'succeeded': succeeded,
        'failed': len(errors), 'failure_rate': len(errors) / len(samples),
        'elapsed_seconds': round(elapsed, 3),
        'papers_per_hour': round(succeeded / elapsed * 3600, 3) if elapsed else 0,
        'latency_p50_seconds': round(statistics.median(latencies), 3) if latencies else None,
        'latency_p95_seconds': _percentile(latencies, 0.95),
        'gpu_peak_memory_mib': max(memory, default=None),
        'gpu_peak_utilization_percent': max(utilization, default=None),
        'errors': {code: errors.count(code) for code in sorted(set(errors))},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', default=','.join(map(str, DEFAULT_LEVELS)))
    parser.add_argument('--sample-size', type=int, default=16)
    parser.add_argument('--model', default=os.environ.get('TOGOS_WSL_LLM_MODEL', DEFAULT_MODEL))
    parser.add_argument('--base-url', default=os.environ.get('TOGOS_WSL_LLM_BASE_URL', 'http://127.0.0.1:8000/v1'))
    parser.add_argument('--timeout', type=float, default=DEFAULT_MODEL_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        levels = tuple(int(value) for value in args.levels.split(','))
        if (not levels or any(not 1 <= value <= MAX_MODEL_WORKERS for value in levels)
                or args.sample_size < max(levels) or args.timeout <= 0 or not args.model.strip()):
            parser.error('levels must be unique workers within range; sample-size must cover the largest level')
        levels = tuple(dict.fromkeys(levels))
        validate_loopback_base_url(args.base_url)
        samples = _cached_samples(args.sample_size)
        results = []
        for workers in levels:
            level_samples = samples_for_level(samples, workers)
            print(f'benchmark workers={workers} samples={len(level_samples)}', flush=True)
            results.append(_run_level(level_samples, workers, args.model, args.base_url, args.timeout))
            print(json.dumps(results[-1], ensure_ascii=False, sort_keys=True), flush=True)
        recommended = recommend_workers(results)
        report = {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                  'model': args.model, 'timeout': args.timeout, 'results': results,
                  'recommended_workers': recommended}
        path = ROOT / 'build' / 'reports' / f"paper-model-benchmark-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        atomic_write_text(path, json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'report': str(path), 'recommended_workers': recommended}, ensure_ascii=False))
        return 0
    except (PaperSummaryError, LoopbackChatError) as error:
        print(f'error {error.code}: {error.message}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
