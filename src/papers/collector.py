"""Collect the public arXiv list on a GitHub-hosted runner."""

from __future__ import annotations

import argparse
import datetime as dt
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
import time
from typing import Callable

import arxiv
import requests
import yaml

from .candidate_ledger import (
    atomic_write_json,
    load_candidate_ledger,
    merge_collected_candidates,
)


logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

ARXIV_BASE_URL = "https://arxiv.org/"
ARXIV_REQUEST_DELAY_SECONDS = 10.0
ARXIV_RETRY_ATTEMPTS = 4
ARXIV_RETRY_BACKOFF_SECONDS = 30
ARXIV_CONNECT_TIMEOUT_SECONDS = 10
ARXIV_READ_TIMEOUT_SECONDS = 60
ARXIV_MAX_INLINE_RETRY_SECONDS = 300
RETRYABLE_ARXIV_STATUSES = {429, 500, 502, 503, 504}


class ArxivRetryExhausted(RuntimeError):
    """Raised after a transient arXiv API failure exhausts its backoff."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after_seconds = retry_after_seconds


def parse_retry_after(value: str | None, *, now: dt.datetime | None = None) -> int | None:
    """Return a nonnegative Retry-After delay from seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0, int(value.strip()))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=dt.timezone.utc)
    current = now or dt.datetime.now(dt.timezone.utc)
    return max(0, math.ceil((retry_at - current).total_seconds()))


class ArxivTimeoutSession(requests.Session):
    """Apply finite connect/read timeouts to the arxiv package session."""

    last_retry_after_seconds: int | None = None

    def request(self, method, url, **kwargs):
        kwargs.setdefault(
            "timeout",
            (ARXIV_CONNECT_TIMEOUT_SECONDS, ARXIV_READ_TIMEOUT_SECONDS),
        )
        response = super().request(method, url, **kwargs)
        self.last_retry_after_seconds = (
            parse_retry_after(response.headers.get("Retry-After"))
            if response.status_code == 429
            else None
        )
        return response


def make_arxiv_client(page_size: int) -> arxiv.Client:
    client = arxiv.Client(
        page_size=page_size,
        delay_seconds=ARXIV_REQUEST_DELAY_SECONDS,
        num_retries=0,
    )
    client._session = ArxivTimeoutSession()
    return client


def is_retryable_arxiv_error(error: Exception) -> bool:
    if isinstance(error, arxiv.HTTPError):
        return error.status in RETRYABLE_ARXIV_STATUSES
    return isinstance(
        error,
        (arxiv.UnexpectedEmptyPageError, requests.ConnectionError, requests.Timeout),
    )


def fetch_arxiv_results(
    client: arxiv.Client, search: arxiv.Search, topic: str
) -> list[arxiv.Result]:
    for attempt in range(1, ARXIV_RETRY_ATTEMPTS + 1):
        try:
            return list(client.results(search))
        except (arxiv.ArxivError, requests.RequestException) as error:
            if not is_retryable_arxiv_error(error):
                raise
            status = getattr(error, "status", None)
            retry_after = getattr(
                getattr(client, "_session", None),
                "last_retry_after_seconds",
                None,
            )
            if not isinstance(retry_after, (int, float)):
                retry_after = None
            wait_seconds = ARXIV_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            if status == 429 and retry_after is not None:
                wait_seconds = max(wait_seconds, math.ceil(retry_after))
            exhausted_now = (
                attempt == ARXIV_RETRY_ATTEMPTS
                or (status == 429 and attempt >= 2)
                or (status == 429 and wait_seconds > ARXIV_MAX_INLINE_RETRY_SECONDS)
            )
            if exhausted_now:
                exhausted = ArxivRetryExhausted(
                    f"arXiv request for {topic!r} failed after "
                    f"{attempt} attempts: {error}",
                    status=status,
                    retry_after_seconds=(
                        wait_seconds if status == 429 else None
                    ),
                )
                raise exhausted from error
            logging.warning(
                "Transient arXiv error for %s (attempt %d/%d): %s; "
                "retrying in %d seconds",
                topic,
                attempt,
                ARXIV_RETRY_ATTEMPTS,
                error,
                wait_seconds,
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def format_query_term(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"' if len(term.split()) > 1 else escaped


def build_filter_query(
    filters: list[str],
    fields: list[str] | None = None,
    categories: list[str] | None = None,
) -> str:
    if not filters:
        raise ValueError("Keyword filters must not be empty")
    if fields:
        scoped_terms = []
        for filter_term in filters:
            formatted_term = format_query_term(filter_term)
            field_query = " OR ".join(
                f"{field}:{formatted_term}" for field in fields
            )
            scoped_terms.append(f"({field_query})")
        filter_query = f'({" OR ".join(scoped_terms)})'
    else:
        filter_query = " OR ".join(format_query_term(term) for term in filters)
    if not categories:
        return filter_query
    category_query = " OR ".join(f"cat:{category}" for category in categories)
    return f"{filter_query} AND ({category_query})"


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["queries"] = {
        topic: build_filter_query(
            settings["filters"],
            fields=settings.get("fields"),
            categories=settings.get("categories"),
        )
        for topic, settings in config["keywords"].items()
    }
    return config


def paper_record(result: arxiv.Result, topic: str) -> dict:
    paper_id = result.get_short_id().split("v", 1)[0]
    title = result.title
    first_author = str(result.authors[0])
    updated = result.updated.date().isoformat()
    paper_url = f"{ARXIV_BASE_URL}abs/{paper_id}"
    pdf_url = (result.pdf_url or f"{ARXIV_BASE_URL}pdf/{paper_id}.pdf").replace(
        "http://", "https://"
    )
    archive_row = (
        f"|**{updated}**|**{title}**|{first_author} et.al.|"
        f"[{paper_id}]({paper_url})|null|\n"
    )
    return {
        "id": paper_id,
        "title": title,
        "abstract": result.summary,
        "authors": f"{first_author} et.al.",
        "updated": updated,
        "paper_url": paper_url,
        "pdf_url": pdf_url,
        "topic": topic,
        "archive_row": archive_row,
    }


def fetch_topic(
    client: arxiv.Client, topic: str, query: str, max_results: int | None = None
) -> list[dict]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    records = [
        paper_record(result, topic)
        for result in fetch_arxiv_results(client, search, topic)
    ]
    for record in records:
        logging.info(
            "Time = %s title = %s author = %s",
            record["updated"],
            record["title"],
            record["authors"],
        )
    return records


def fetch_collection(
    config: dict,
    ledger: dict,
    *,
    now: dt.datetime | None = None,
    clock: Callable[[], dt.datetime] | None = None,
    skip_current_topics: bool = False,
    checkpoint: Callable[
        [list[dict], dict[str, str], dict[str, dict], str | None], None
    ]
    | None = None,
) -> tuple[list[dict], list[str], dict[str, str]]:
    """Collect topic shards, exposing each recoverable checkpoint to the caller."""
    settings = config.get("collection", {})
    lookback_days = max(1, int(settings.get("lookback_days", 7)))
    overlap_days = max(2, int(settings.get("overlap_days", 2)))
    batch_size = max(1, int(settings.get("query_batch_size", 8)))
    page_size = min(2000, max(1, int(settings.get("page_size", 100))))
    end = now or dt.datetime.now(dt.timezone.utc)
    if end.tzinfo is None:
        raise ValueError("Collection time must include a timezone")
    end = end.astimezone(dt.timezone.utc).replace(microsecond=0)
    current_time = clock or (lambda: dt.datetime.now(dt.timezone.utc))
    cursors = dict(ledger.get("collection_cursors", {}))
    topics = list(config["keywords"])
    raw_progress = ledger.get("collection_progress", {})
    if isinstance(raw_progress, dict):
        progress = {
            topic: dict(value)
            for topic, value in raw_progress.items()
            if topic in topics and isinstance(value, dict)
        }
    else:
        progress = {}
    records: list[dict] = []
    failed_topics: list[str] = []

    cooldown_until = ledger.get("collection_cooldown_until")
    if cooldown_until is not None:
        cooldown = _parse_utc_timestamp(cooldown_until, "collection cooldown")
        if cooldown > end:
            return (
                [*records],
                [
                    topic
                    for topic in topics
                    if not (
                        skip_current_topics
                        and _cursor_is_current(cursors.get(topic), end.date())
                    )
                ],
                cursors,
            )
        cooldown_until = None
        if checkpoint is not None:
            checkpoint([], cursors, progress, None)

    client = make_arxiv_client(page_size)
    rate_limited = False
    for topic, keyword_settings in config["keywords"].items():
        if skip_current_topics and _cursor_is_current(cursors.get(topic), end.date()):
            continue
        if rate_limited:
            failed_topics.append(topic)
            continue
        default_start = end - dt.timedelta(days=lookback_days)
        if topic in cursors:
            cursor = _parse_utc_timestamp(cursors[topic], f"cursor for {topic!r}")
            if cursor > end:
                raise ValueError(f"Invalid collection cursor for {topic!r}")
            default_start = cursor - dt.timedelta(days=overlap_days)
        filters = keyword_settings["filters"]
        if not filters:
            raise ValueError(f"Keyword filters must not be empty: {topic}")
        signature = _query_plan_signature(keyword_settings, batch_size)
        topic_progress = _valid_topic_progress(
            progress.get(topic), signature=signature, latest_end=end
        )
        if topic_progress is None:
            topic_progress = {
                "window_start": default_start.isoformat(),
                "window_end": end.isoformat(),
                "query_signature": signature,
                "completed_shards": [],
            }
            progress[topic] = topic_progress
        start = _parse_utc_timestamp(
            topic_progress["window_start"], f"window start for {topic!r}"
        )
        topic_end = _parse_utc_timestamp(
            topic_progress["window_end"], f"window end for {topic!r}"
        )
        window = f"submittedDate:[{start:%Y%m%d%H%M} TO {topic_end:%Y%m%d%H%M}]"
        shard_count = math.ceil(len(filters) / batch_size)
        completed_shards = {
            shard
            for shard in topic_progress.get("completed_shards", [])
            if isinstance(shard, int)
            and not isinstance(shard, bool)
            and 0 <= shard < shard_count
        }
        topic_progress["completed_shards"] = sorted(completed_shards)
        topic_failed = False
        for shard_index, offset in enumerate(range(0, len(filters), batch_size)):
            if shard_index in completed_shards:
                continue
            try:
                query = build_filter_query(
                    filters[offset : offset + batch_size],
                    fields=keyword_settings.get("fields"),
                    categories=keyword_settings.get("categories"),
                )
                shard_records = fetch_topic(client, topic, f"({query}) AND {window}")
            except (ArxivRetryExhausted, arxiv.ArxivError, requests.RequestException) as error:
                logging.error("Preserving shard progress for failed topic %s: %s", topic, error)
                failed_topics.append(topic)
                topic_failed = True
                rate_limited = getattr(error, "status", None) == 429
                if rate_limited:
                    retry_after = getattr(error, "retry_after_seconds", None)
                    if not isinstance(retry_after, (int, float)):
                        retry_after = ARXIV_RETRY_BACKOFF_SECONDS * 2
                    cooldown_start = current_time()
                    if cooldown_start.tzinfo is None:
                        raise ValueError("Cooldown clock must include a timezone")
                    cooldown_start = cooldown_start.astimezone(dt.timezone.utc)
                    cooldown_until = (
                        cooldown_start
                        + dt.timedelta(seconds=max(1, math.ceil(retry_after)))
                    ).isoformat()
                if checkpoint is not None:
                    checkpoint([], cursors, progress, cooldown_until)
                break
            records.extend(shard_records)
            completed_shards.add(shard_index)
            topic_progress["completed_shards"] = sorted(completed_shards)
            if checkpoint is not None:
                checkpoint(shard_records, cursors, progress, cooldown_until)
        if topic_failed:
            continue
        if len(completed_shards) == shard_count:
            cursors[topic] = topic_end.isoformat()
            progress.pop(topic, None)
            if checkpoint is not None:
                checkpoint([], cursors, progress, cooldown_until)

    stale_topics = [
        topic
        for topic in topics
        if not _cursor_is_current(cursors.get(topic), end.date())
    ]
    if not failed_topics and stale_topics:
        resume_ledger = {
            **ledger,
            "collection_cursors": cursors,
            "collection_progress": progress,
        }
        if cooldown_until is None:
            resume_ledger.pop("collection_cooldown_until", None)
        catchup_records, catchup_failures, cursors = fetch_collection(
            config,
            resume_ledger,
            now=end,
            clock=clock,
            skip_current_topics=True,
            checkpoint=checkpoint,
        )
        records.extend(catchup_records)
        failed_topics.extend(catchup_failures)
    return records, failed_topics, cursors


def _parse_utc_timestamp(value: object, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError(f"Invalid {label}")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Invalid {label}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"Invalid {label}")
    return parsed.astimezone(dt.timezone.utc)


def _cursor_is_current(value: object, day: dt.date) -> bool:
    if value is None:
        return False
    return _parse_utc_timestamp(value, "collection cursor").date() == day


def _query_plan_signature(settings: dict, batch_size: int) -> str:
    plan = {
        "filters": settings.get("filters", []),
        "fields": settings.get("fields"),
        "categories": settings.get("categories"),
        "batch_size": batch_size,
    }
    encoded = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_topic_progress(
    value: dict | None, *, signature: str, latest_end: dt.datetime
) -> dict | None:
    if not value or value.get("query_signature") != signature:
        return None
    try:
        start = _parse_utc_timestamp(value.get("window_start"), "collection window")
        end = _parse_utc_timestamp(value.get("window_end"), "collection window")
    except ValueError:
        return None
    if start > end or end > latest_end:
        return None
    completed = value.get("completed_shards")
    if not isinstance(completed, list):
        return None
    return {
        **value,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "completed_shards": list(completed),
    }


def fetch_all_topics(queries: dict[str, str], max_results: int) -> tuple[list, list]:
    client = make_arxiv_client(min(max(max_results, 1), 100))
    records: list[dict] = []
    failed_topics: list[str] = []
    for topic, query in queries.items():
        logging.info("Keyword: %s", topic)
        try:
            records.extend(fetch_topic(client, topic, query, max_results))
        except ArxivRetryExhausted as error:
            logging.error("Skipping topic after retries: %s", error)
            failed_topics.append(topic)
    if failed_topics and len(failed_topics) == len(queries):
        raise RuntimeError(
            "All arXiv topics failed after retries: " + ", ".join(failed_topics)
        )
    if failed_topics:
        logging.warning(
            "Completed with stale data preserved for failed topics: %s",
            ", ".join(failed_topics),
        )
    return records, failed_topics


def load_archive(path: str | Path) -> dict:
    archive_path = Path(path)
    if not archive_path.exists():
        return {}
    content = archive_path.read_text(encoding="utf-8")
    return json.loads(content) if content else {}


def collect(
    config_path: str | Path,
    *,
    skip_if_current: bool = False,
    now: dt.datetime | None = None,
    clock: Callable[[], dt.datetime] | None = None,
) -> dict:
    config = load_config(config_path)
    archive_path = config["json_gitpage_path"]
    html_path = config["html_gitpage_path"]
    output_root = config.get("output_root", str(Path(html_path).parent))
    search_index_path = config.get(
        "search_index_path", str(Path(output_root) / "search-index.json")
    )
    ledger_path = config["candidate_ledger_path"]
    milestone_catalog_path = config["milestone_catalog_path"]
    topics = list(config["queries"])
    archive = load_archive(archive_path)
    ledger = load_candidate_ledger(ledger_path)
    end = now or dt.datetime.now(dt.timezone.utc)
    if end.tzinfo is None:
        raise ValueError("Collection time must include a timezone")
    end = end.astimezone(dt.timezone.utc).replace(microsecond=0)
    today = end.date()
    if skip_if_current and all(
        _cursor_is_current(ledger.get("collection_cursors", {}).get(topic), today)
        for topic in topics
    ):
        return {
            "status": "already_current",
            "collected": 0,
            "new_candidates": 0,
            "failed_topics": 0,
            "failed_topic_names": [],
        }

    working_ledger = ledger
    added = 0

    def save_checkpoint(
        shard_records: list[dict],
        cursors: dict[str, str],
        progress: dict[str, dict],
        cooldown_until: str | None,
    ) -> None:
        nonlocal working_ledger, added
        _, next_ledger, newly_added = merge_collected_candidates(
            archive, working_ledger, shard_records, topics
        )
        added += newly_added
        next_ledger["collection_cursors"] = dict(cursors)
        if progress:
            next_ledger["collection_progress"] = progress
        else:
            next_ledger.pop("collection_progress", None)
        if cooldown_until is not None:
            next_ledger["collection_cooldown_until"] = cooldown_until
        else:
            next_ledger.pop("collection_cooldown_until", None)
        atomic_write_json(ledger_path, next_ledger)
        working_ledger = next_ledger

    initial_cooldown = ledger.get("collection_cooldown_until")
    cooldown_was_active = (
        initial_cooldown is not None
        and _parse_utc_timestamp(initial_cooldown, "collection cooldown") > end
    )
    records, failed_topics, _ = fetch_collection(
        config,
        ledger,
        now=end,
        clock=clock,
        skip_current_topics=skip_if_current,
        checkpoint=save_checkpoint,
    )
    status = "complete"
    if failed_topics:
        status = "cooldown" if cooldown_was_active else "incomplete"
    result = {
        "collected": len(records),
        "new_candidates": added,
        "failed_topics": len(failed_topics),
        "failed_topic_names": failed_topics,
        "status": status,
    }
    cooldown_until = working_ledger.get("collection_cooldown_until")
    if cooldown_until is not None:
        result["cooldown_until"] = cooldown_until
    logging.info("Cloud candidate collection result = %s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/site.yaml")
    parser.add_argument('--report', type=Path, help='write collection status JSON')
    parser.add_argument('--skip-if-current', action='store_true')
    args = parser.parse_args()
    result = collect(args.config, skip_if_current=args.skip_if_current)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.report, result)
    if result['failed_topics']:
        logging.error('Collection incomplete; saved completed topics and preserved failed cursors. Next scheduled catch-up will retry: %s', ', '.join(result['failed_topic_names']))
        sys.exit(75)


if __name__ == "__main__":
    main()
