"""CLI for local paper annotation and coverage checks."""

from __future__ import annotations
from papers.model_runtime import DEFAULT_MODEL, DEFAULT_MODEL_TIMEOUT_SECONDS, DEFAULT_MODEL_WORKERS

import argparse
import json
import os
import sys

from .models import PaperAnnotationError
from .workflow import run_annotations, status_snapshot


DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m papers.annotations")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="classify newly accepted papers; --paper explicitly selects an archived paper")
    run.add_argument("--model", default=os.environ.get("TOGOS_WSL_LLM_MODEL", DEFAULT_MODEL))
    run.add_argument("--base-url", default=os.environ.get("TOGOS_WSL_LLM_BASE_URL", DEFAULT_BASE_URL))
    run.add_argument("--timeout", type=float, default=DEFAULT_MODEL_TIMEOUT_SECONDS)
    run.add_argument("--workers", type=int, default=os.environ.get("TOGOS_WSL_LLM_WORKERS", str(DEFAULT_MODEL_WORKERS)))
    run.add_argument("--paper", action="append", default=[])
    run.add_argument("--limit", type=int, default=100)
    run.add_argument("--refresh", action="store_true")
    status = commands.add_parser("status", help="show archive annotation coverage")
    status.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _execute(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        snapshot = status_snapshot()
        if args.as_json:
            print(json.dumps(snapshot, sort_keys=True))
        else:
            print(f"total={snapshot['total']} annotated={snapshot['annotated']} pending={snapshot['pending']}")
        return 0
    result = run_annotations(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        workers=args.workers,
        paper_ids=tuple(args.paper),
        limit=args.limit,
        refresh=args.refresh,
    )
    print(f"selected={result.selected} succeeded={result.succeeded} failed={result.failed}")
    for record in result.records:
        if record.status == "failed":
            print(f"failed {record.arxiv_id} {record.error_code}: {record.error_message}", file=sys.stderr)
    return 3 if result.partial else 0


def main() -> None:
    try:
        raise SystemExit(_execute())
    except PaperAnnotationError as error:
        print(f"error {error.code}: {error.message}", file=sys.stderr)
        raise SystemExit(2) from None
