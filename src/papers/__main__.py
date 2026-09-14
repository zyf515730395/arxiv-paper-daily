"""Single entry point for collection, review, batches, publication and status."""
from __future__ import annotations

import argparse
import json
import os
import sys

from papers import paths
from papers.model_runtime import DEFAULT_MODEL, DEFAULT_MODEL_TIMEOUT_SECONDS, DEFAULT_MODEL_WORKERS, MAX_MODEL_WORKERS


def build():
    from papers.site import generate_site
    from shared.rendering import atomic_write_bytes
    generate_site(paths.ARCHIVE, paths.DOCS / 'index.html', paths.LEDGER,
                  paths.ROOT / 'config/milestone_models.yaml', output_root=paths.DOCS,
                  search_index_path=paths.DOCS / 'search-index.json', config_path=paths.CONFIG,
                  annotation_path=paths.ANNOTATIONS, refresh_related=False)
    atomic_write_bytes(paths.DOCS / 'togos-papers.json', paths.ARCHIVE.read_bytes())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'recheck':
        from papers.recheck import main as recheck_main
        return recheck_main(argv[1:])
    if argv and argv[0] == 'batch':
        from papers.batch.cycle import main as batch_main
        return batch_main(argv[1:])
    if argv and argv[0] == 'benchmark':
        from papers.batch.benchmark import main as benchmark_main
        return benchmark_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'curate', 'build', 'status', 'daily', 'publish-offline'])
    parser.add_argument('--model', default=os.environ.get('TOGOS_WSL_LLM_MODEL', DEFAULT_MODEL))
    parser.add_argument('--base-url', default=os.environ.get('TOGOS_WSL_LLM_BASE_URL', 'http://127.0.0.1:8000/v1'))
    parser.add_argument('--timeout', type=float, default=DEFAULT_MODEL_TIMEOUT_SECONDS)
    parser.add_argument('--workers', type=int, default=DEFAULT_MODEL_WORKERS)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.limit < 1 or not 1 <= args.workers <= MAX_MODEL_WORKERS or args.timeout <= 0:
        parser.error(f'limit/timeout must be positive; workers must be 1-{MAX_MODEL_WORKERS}')
    if args.command == 'collect':
        from papers.collector import collect
        print(json.dumps(collect(paths.CONFIG)))
    elif args.command == 'build':
        if not args.dry_run:
            build()
    elif args.command == 'status':
        from papers.summaries.workflow import status_snapshot
        from papers.curation import run
        print(json.dumps({'curation': run(model=args.model, base_url=args.base_url, dry_run=True),
                          'summaries': status_snapshot()}, ensure_ascii=False))
    elif args.command == 'publish-offline':
        from papers.summaries.offline import publish_offline_summaries
        from papers.batch.publisher import publish_annotations
        print(publish_offline_summaries(dry_run=args.dry_run))
        print(publish_annotations(dry_run=args.dry_run))
        if not args.dry_run:
            build()
    else:
        from papers.curation import run
        result = run(model=args.model, base_url=args.base_url, timeout=args.timeout, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False))
        partial = bool(result.get('failures'))
        if args.command == 'daily' and not args.dry_run:
            from papers.summaries.workflow import run_summaries
            from papers.annotations.workflow import run_annotations
            summary = run_summaries(model=args.model, base_url=args.base_url, timeout=args.timeout,
                                    workers=args.workers, limit=args.limit)
            annotation = run_annotations(model=args.model, base_url=args.base_url, timeout=args.timeout,
                                         workers=args.workers, limit=args.limit)
            print(f'summaries={summary.succeeded}/{summary.selected} annotations={annotation.succeeded}/{annotation.selected}')
            partial = partial or bool(summary.failed or annotation.failed)
        if not args.dry_run:
            build()
        return 3 if partial else 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
