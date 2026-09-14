"""Private two-stage batch adapter. Never publishes pages or changes Git state."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from uuid import uuid4

from papers.paths import ARCHIVE, CONFIG, DOCS, LEDGER
from papers.model_runtime import DEFAULT_MODEL, DEFAULT_MODEL_TIMEOUT_SECONDS, DEFAULT_MODEL_WORKERS, MAX_MODEL_WORKERS
from papers.summaries.acquisition import ArxivSourceClient
from papers.annotations.catalog import (
    annotation_labels_for_topics,
    annotation_value,
    load_annotation_definitions,
    load_topic_tag_allowlists,
    load_topic_tag_dimensions,
)
from papers.annotations.classifier import classify_paper, taxonomy_hash
from papers.annotations.models import PaperAnnotationError
from papers.annotations.prompts import PROMPT_VERSION as ANNOTATION_PROMPT_VERSION
from papers.summaries.cache import PaperSummaryCache, cache_key
from papers.summaries.catalog import TOPIC_SLUGS, notes_path
from papers.summaries.extraction import EXTRACTION_VERSION
from papers.summaries.models import PaperSummaryError
from papers.summaries.paths import private_path, run_lock
from papers.summaries.prompts import PROMPT_VERSION, build_chunks
from papers.summaries.summarizer import summarize_paper
from shared.loopback_chat import LoopbackChatError, validate_loopback_base_url
from shared.rendering import atomic_write_text
from .catalog import archive_candidates
from .review import POLICY_VERSION, RULES, review_topics

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE = ARCHIVE
DEFAULT_LEDGER = LEDGER
DEFAULT_DOCS = DOCS
DEFAULT_CONFIG = CONFIG
PAPER_LABELS = load_annotation_definitions(DEFAULT_CONFIG)
TOPIC_TAG_DIMENSIONS = load_topic_tag_dimensions(DEFAULT_CONFIG, PAPER_LABELS)
TOPIC_TAG_ALLOWLISTS = load_topic_tag_allowlists(DEFAULT_CONFIG, PAPER_LABELS)
TOPIC_REVIEW_ACTION = "remove_rejected_topic_entries"


def annotation_policy_hash():
    payload = {
        "taxonomy": taxonomy_hash(PAPER_LABELS),
        "topic_tag_dimensions": TOPIC_TAG_DIMENSIONS,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


ANNOTATION_POLICY_HASH = annotation_policy_hash()


def note_paths(item):
    folder = TOPIC_SLUGS[item.topic]
    return (private_path("markdown", folder, f"{item.arxiv_id}.md"),
            private_path("markdown", folder, f"{item.arxiv_id}.json"))


def note_identity(item):
    return {"id": item.arxiv_id, "title": item.title, "topic": item.topic,
            "updated": item.updated.isoformat(), "prompt_version": PROMPT_VERSION,
            "extraction_version": EXTRACTION_VERSION}


def review_identity(item):
    payload = {"historical": item.historical, "review_state": item.review_state,
               "abstract": item.abstract, "policy": POLICY_VERSION, "rules": RULES}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def note_status(item):
    note, receipt = note_paths(item)
    if not note.exists():
        return "missing"
    try:
        if note.stat().st_size > 128 * 1024 or receipt.stat().st_size > 16 * 1024:
            return "conflict"
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        identity = note_identity(item)
        if (not isinstance(metadata, dict) or metadata.get("version") != 1
                or any(metadata.get(key) != value for key, value in identity.items())
                or metadata.get("markdown_sha256") != hashlib.sha256(note.read_bytes()).hexdigest()):
            return "conflict"
        if (metadata.get("annotation_schema_version") != 2
                or metadata.get("annotation_prompt_version") != ANNOTATION_PROMPT_VERSION
                or metadata.get("annotation_taxonomy_hash") != ANNOTATION_POLICY_HASH):
            return "needs_review"
        if ("accept" not in metadata or metadata.get("review_policy") != POLICY_VERSION
                or metadata.get("review_input") != review_identity(item)):
            return "needs_review"
        if (metadata["accept"] is not None and type(metadata["accept"]) is not bool
                or not isinstance(metadata.get("accept_reason"), str)):
            return "conflict"
        return "ready"
    except (OSError, ValueError, UnicodeError):
        return "conflict"


def save_note(item, summary, source, model, decision, annotation):
    note, receipt = note_paths(item)
    if note_status(item) not in {"missing", "needs_review"}:
        raise PaperSummaryError("local_note_conflict", "existing Markdown was preserved")
    contributions = "\n".join(f"- {value}" for value in summary.contributions)
    markdown = (
        f"# {item.title}\n\n"
        f"- arXiv: https://arxiv.org/abs/{item.arxiv_id}\n"
        f"- 主题：{item.topic}\n- 更新日期：{item.updated.isoformat()}\n"
        f"- 依据：Introduction（{source.kind.upper()}）\n\n"
        f"## 一句话总结\n\n{summary.one_sentence}\n\n"
        f"## 解决的问题\n\n{summary.problem}\n\n"
        f"## 创新点\n\n{contributions}\n\n"
        f"## 主题适用性（本地审阅）\n\n"
        f"- accept: {json.dumps(decision['accept'])}\n"
        f"- 历史归档：{'是' if item.historical else '否'}\n"
        f"- 判断理由：{decision['reason']}\n"
        f"- 判断来源：{decision['origin']}\n"
        f"\n## 论文分类\n\n"
        f"- 主题：{', '.join(annotation.topics)}\n"
        f"- 标签：{', '.join(annotation.tags)}\n"
        f"- 类型：{annotation.paper_type}\n"
        f"- 机构：{', '.join(annotation.institutions) or '-'}\n"
    )
    metadata = {"version": 1, **note_identity(item), "model": model,
                "historical": item.historical, "accept": decision['accept'],
                "accept_reason": decision['reason'], "review_origin": decision['origin'],
                "review_policy": POLICY_VERSION,
                "review_input": review_identity(item),
                **annotation_value(annotation), "annotation_schema_version": 2,
                "annotation_prompt_version": ANNOTATION_PROMPT_VERSION,
                "annotation_taxonomy_hash": ANNOTATION_POLICY_HASH,
                "source_sha256": source.source_sha256,
                "source": source.source_path.relative_to(private_path()).as_posix(),
                "summary_cache": PaperSummaryCache().path_for(
                    cache_key(source.source_sha256, model)
                ).relative_to(private_path()).as_posix(),
                "public_url": notes_path(DEFAULT_DOCS, item.topic).relative_to(
                    DEFAULT_DOCS
                ).as_posix() + f"#summary-{item.arxiv_id}",
                "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest()}
    # Receipt first: a crash before Markdown creation remains safely retryable.
    atomic_write_text(receipt, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(note, markdown)
    return f"markdown/{TOPIC_SLUGS[item.topic]}/{item.arxiv_id}.md"


def process(items, args):
    item = items[0]
    factory = getattr(args, "download_session_factory", None) if args.mode == "download" else None
    client = ArxivSourceClient(session=factory() if factory else None)
    if args.mode == "download":
        try:
            source = client.acquire(item.arxiv_id, item.title)
        finally:
            client.session.close()
        return source, None, None, None
    # Validates and loads the HTML/PDF extraction cache; never falls back to HTTP.
    source = client._load_cached(item.arxiv_id, item.title)
    if source is None:
        raise PaperSummaryError("source_cache_missing_or_invalid",
                                "run the batch download stage while online first")
    build_chunks(source.document)
    summary = summarize_paper(source, model=args.model, base_url=args.base_url,
                              timeout=args.timeout, refresh=False)
    decisions = review_topics(items, source, args.model, args.base_url, args.timeout)
    annotation = classify_paper(
        source, annotation_labels_for_items(items), model=args.model, base_url=args.base_url,
        timeout=args.timeout, refresh=False,
    )
    return source, summary, decisions, annotation


def annotation_labels_for_items(items):
    topics = tuple(dict.fromkeys(item.topic for item in items))
    return annotation_labels_for_topics(PAPER_LABELS, TOPIC_TAG_ALLOWLISTS, topics)


def existing_note_keys():
    # Directory snapshots avoid resolving 16k nonexistent paths on WSL-mounted drives.
    result = set()
    for topic, slug in TOPIC_SLUGS.items():
        folder = private_path("markdown", slug)
        if folder.is_dir():
            result.update((topic, path.stem) for path in folder.iterdir() if path.suffix == ".md")
    return result


def select_items(args):
    if not DEFAULT_LEDGER.is_file() or not DEFAULT_ARCHIVE.is_file():
        raise PaperSummaryError("input_missing", "repository archive or candidate ledger is missing")
    candidates = archive_candidates(DEFAULT_ARCHIVE, DEFAULT_LEDGER, tuple(args.paper))
    selected = []
    skipped = 0
    source_ready = {}
    existing_notes = existing_note_keys()
    source_root = private_path("sources")
    existing_sources = {path.name for path in source_root.iterdir()} if source_root.is_dir() else set()
    for item in candidates:
        state = note_status(item) if (item.topic, item.arxiv_id) in existing_notes else "missing"
        if state == "ready":
            skipped += 1
            continue
        if args.mode == "download" and state != "conflict" and item.arxiv_id in existing_sources:
            key = (item.arxiv_id, item.title)
            if key not in source_ready:
                source_ready[key] = ArxivSourceClient()._load_cached(*key) is not None
            if source_ready[key]:
                skipped += 1
                continue
        selected.append((item, state))
    if args.limit is not None:
        ids = set(list(dict.fromkeys(item.arxiv_id for item, _ in selected))[:args.limit])
        selected = [(item, state) for item, state in selected if item.arxiv_id in ids]
    return selected, skipped


def context_candidates(paper_ids):
    """Load every archived topic for explicitly selected papers, including public-ready ones."""
    requested = tuple(paper_ids)
    if not requested:
        return {}
    result = {}
    for item in archive_candidates(DEFAULT_ARCHIVE, DEFAULT_LEDGER, requested):
        result.setdefault(item.arxiv_id, []).append(item)
    return result


def write_topic_review():
    """Aggregate verified local receipts, including successes from earlier batches."""
    result = {"version": 1, "policy_version": POLICY_VERSION,
              "action": TOPIC_REVIEW_ACTION, "accept_candidates": [],
              "reject_candidates": [], "needs_review": []}
    existing_notes = existing_note_keys()
    for item in archive_candidates(DEFAULT_ARCHIVE, DEFAULT_LEDGER, include_ready=True):
        if (item.topic, item.arxiv_id) not in existing_notes or note_status(item) != "ready":
            continue
        _, receipt = note_paths(item)
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        key = ("accept_candidates" if metadata["accept"] is True else
               "reject_candidates" if metadata["accept"] is False else "needs_review")
        result[key].append({name: metadata[name] for name in (
            "id", "topic", "title", "historical", "accept", "accept_reason",
            "review_origin", "source", "summary_cache", "public_url")})
    path = private_path("batch", "topic-review.json")
    atomic_write_text(path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return str(path)


@contextmanager
def batch_executor(workers):
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        yield executor
    except KeyboardInterrupt:
        print("Stopping: canceling queued papers; waiting only for active workers to finish safely.",
              file=sys.stderr, flush=True)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def run(args, *, report_sink=None):
    selected, skipped = select_items(args)
    unique_count = len({item.arxiv_id for item, _ in selected})
    print(f"mode={args.mode} selected={len(selected)} unique_papers={unique_count} "
          f"completed_skipped={skipped}", flush=True)
    if args.dry_run:
        for item, state in selected:
            cached = ArxivSourceClient()._load_cached(item.arxiv_id, item.title) is not None
            print(f"{item.arxiv_id} topic={item.topic} historical={item.historical} "
                  f"cache={'ready' if cached else 'missing'} local={state}")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = private_path("batch", f"{args.mode}-{stamp}-{uuid4().hex[:8]}.json")
    records = [{"id": item.arxiv_id, "topic": item.topic, "historical": item.historical, "status": "pending"}
               for item, _ in selected]
    report = {"version": 1, "mode": args.mode, "status": "running",
              "selected": len(selected), "unique_papers": unique_count, "completed_skipped": skipped,
              "workers": min(args.workers, unique_count),
              "records": records}

    def persist():
        atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if report_sink is not None:
            report_sink.update(report, report_path=str(report_path))

    persist()
    groups = {}
    for index, (item, state) in enumerate(selected):
        if state == "conflict":
            records[index].update(status="failed", error="local_note_conflict")
            print(f"{item.arxiv_id} failed local_note_conflict (preserved)", flush=True)
        else:
            groups.setdefault(item.arxiv_id, []).append((index, item))
    # Include already completed siblings in the judgment context after partial failures.
    contexts = context_candidates(groups)
    with batch_executor(args.workers) as executor:
        futures = {executor.submit(process, contexts[paper_id], args): group
                   for paper_id, group in groups.items()}
        for completed, future in enumerate(as_completed(futures), 1):
            group = futures.pop(future)  # Release large extracted documents as work completes.
            try:
                source, summary, decisions, annotation = future.result()
            except (PaperSummaryError, PaperAnnotationError, LoopbackChatError) as error:
                for index, _ in group:
                    records[index].update(status="failed", error=error.code)
            except Exception:
                for index, _ in group:
                    records[index].update(status="failed", error="unexpected_failure")
            else:
                for index, item in group:
                    try:
                        path = (save_note(item, summary, source, args.model, decisions[item.topic], annotation)
                                if args.mode == "summarize" else f"sources/{item.arxiv_id}/source.{source.kind}")
                        records[index].update(status="succeeded", source=source.kind, output=path)
                        if decisions is not None:
                            records[index].update(accept=decisions[item.topic]['accept'],
                                                  accept_reason=decisions[item.topic]['reason'])
                    except PaperSummaryError as error:
                        records[index].update(status="failed", error=error.code)
                    except Exception:
                        records[index].update(status="failed", error="local_note_write_failed")
            for index, item in group:
                print(f"{item.arxiv_id} [{item.topic}] {records[index]['status']} "
                      f"{records[index].get('error', records[index].get('output', ''))}", flush=True)
            # Each source/cache/note is durable immediately; avoid rewriting an 8k-row report per paper.
            if completed % 20 == 0:
                persist()
    failed = sum(item["status"] == "failed" for item in records)
    report.update(status="complete", succeeded=len(records) - failed, failed=failed)
    if args.mode == "summarize":
        report['topic_review'] = write_topic_review()
    persist()
    print(f"succeeded={len(records)-failed} failed={failed} report={report_path}", flush=True)
    if any(item.get("error") == "model_unavailable" for item in records):
        print("Local model unavailable: start your service (on this host: sudo systemctl start vllm-paper.service), "
              "wait for /v1/models, then rerun. Successful papers remain saved.", file=sys.stderr)
    return 3 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("download", "summarize"))
    parser.add_argument("--workers", type=int,
                        default=os.environ.get("TOGOS_WSL_LLM_WORKERS", str(DEFAULT_MODEL_WORKERS)))
    parser.add_argument("--limit", type=int, help="maximum unique papers; default is all unfinished archive papers across all years")
    parser.add_argument("--paper", action="append", default=[], help="archived arXiv ID, repeatable; includes every missing topic")
    parser.add_argument("--dry-run", action="store_true", help="list candidates; no download, inference or writes")
    parser.add_argument("--model", default=os.environ.get("TOGOS_WSL_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("TOGOS_WSL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--timeout", type=float, default=DEFAULT_MODEL_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.workers <= MAX_MODEL_WORKERS:
            raise PaperSummaryError("invalid_workers", f"workers must be 1-{MAX_MODEL_WORKERS}")
        if args.limit is not None and args.limit < 1:
            raise PaperSummaryError("invalid_limit", "limit must be positive")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise PaperSummaryError("invalid_timeout", "timeout must be finite and positive")
        if args.mode == "summarize":
            validate_loopback_base_url(args.base_url)
            if not args.model.strip():
                raise PaperSummaryError("model_required", "provide a local model name")
        if args.dry_run:
            return run(args)
        with run_lock():
            return run(args)
    except (PaperSummaryError, LoopbackChatError) as error:
        print(f"error {error.code}: {error.message}", file=sys.stderr)
        return 2
    except OSError:
        print("error local_io_failed: local input or report unavailable; preserve state and inspect paths", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted: completed sources/summary caches are retained; rerun to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
