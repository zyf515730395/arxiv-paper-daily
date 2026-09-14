"""Validate and atomically extend the existing public paper-note pages."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re

from shared.rendering import atomic_write_bytes, atomic_write_text

from .catalog import TOPIC_SLUGS, PaperCandidate, notes_path
from .models import PaperSummary, PaperSummaryError


LIST_MARKER = '    <div class="summary-topic-list">\n'
MANIFEST_PATTERN = re.compile(
    r'(?P<prefix><script type="application/json" id="summary-catalog">)'
    r'(?P<body>.*?)(?P<suffix></script>)',
    re.DOTALL,
)
ARTICLE_PATTERN = re.compile(
    r'    <article class="summary-article summary-topic-entry" '
    r'id="summary-(?P<id>\d{4}\.\d{4,5})" data-arxiv-id="(?P=id)" '
    r'data-status="ready">\n.*?</article>\n',
    re.DOTALL,
)


def restore_topic_document(docs_root: Path, topic: str, original: bytes | None) -> None:
    """Restore an existing page or remove only a newly created topic page."""
    path = notes_path(docs_root, topic)
    if original is None:
        if path.resolve().parent != (docs_root / "notes").resolve():
            raise PaperSummaryError("unsafe_summary_path", "summary rollback escaped notes directory")
        path.unlink(missing_ok=True)
    else:
        atomic_write_bytes(path, original)


def _manifest(document: str, path: Path) -> tuple[re.Match[str], dict]:
    match = MANIFEST_PATTERN.search(document)
    if match is None:
        raise PaperSummaryError("invalid_summary_page", f"summary manifest missing: {path.name}")
    try:
        payload = json.loads(match.group("body"))
    except json.JSONDecodeError:
        raise PaperSummaryError("invalid_summary_page", f"summary manifest is invalid: {path.name}") from None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"version", "topic", "papers"}
        or payload["version"] != 1
        or not isinstance(payload["topic"], str)
        or not isinstance(payload["papers"], dict)
    ):
        raise PaperSummaryError("invalid_summary_page", f"summary manifest schema is invalid: {path.name}")
    return match, payload


def _validate(document: str, path: Path, expected_topic: str) -> tuple[dict, dict[str, str]]:
    if document.count(LIST_MARKER) != 1:
        raise PaperSummaryError("invalid_summary_page", f"summary list marker is invalid: {path.name}")
    _, manifest = _manifest(document, path)
    if manifest["topic"] != expected_topic:
        raise PaperSummaryError("invalid_summary_page", f"summary topic mismatch: {path.name}")
    articles: dict[str, str] = {}
    for match in ARTICLE_PATTERN.finditer(document):
        paper_id = match.group("id")
        if paper_id in articles:
            raise PaperSummaryError("duplicate_summary", f"duplicate paper summary: {paper_id}")
        articles[paper_id] = match.group(0)
    ready_ids = {
        paper_id
        for paper_id, value in manifest["papers"].items()
        if isinstance(value, dict) and value.get("status") == "ready"
    }
    if ready_ids != set(articles):
        raise PaperSummaryError("invalid_summary_page", f"manifest and articles disagree: {path.name}")
    for paper_id, value in manifest["papers"].items():
        if not isinstance(value, dict) or value.get("status") not in {"pending", "ready"}:
            raise PaperSummaryError("invalid_summary_page", f"invalid manifest paper: {paper_id}")
        if value.get("status") == "ready":
            expected = f"notes/{path.name}#summary-{paper_id}"
            if value.get("url") != expected:
                raise PaperSummaryError("invalid_summary_page", f"invalid summary URL: {paper_id}")
    return manifest, articles


def load_ready_keys(
    docs_root: str | Path, *, strict: bool = True
) -> set[tuple[str, str]]:
    ready: set[tuple[str, str]] = set()
    notes = Path(docs_root) / "notes"
    managed_names = {f'{slug}.html' for slug in TOPIC_SLUGS.values()}
    for path in sorted(notes.glob("*.html")):
        if path.name not in managed_names:
            continue
        try:
            expected_topic = next(
                topic
                for topic, slug in TOPIC_SLUGS.items()
                if path.name == f"{slug}.html"
            )
            document = path.read_text(encoding="utf-8")
            _, manifest = _manifest(document, path)
            _validate(document, path, expected_topic)
        except (OSError, UnicodeError, StopIteration, PaperSummaryError) as error:
            if strict:
                if isinstance(error, PaperSummaryError):
                    raise
                raise PaperSummaryError(
                    "invalid_summary_page",
                    f"summary page cannot be validated: {path.name}",
                ) from None
            continue
        for paper_id, value in manifest["papers"].items():
            if isinstance(value, dict) and value.get("status") == "ready":
                ready.add((manifest["topic"], paper_id))
    return ready


def load_ready_ids(docs_root: str | Path, *, strict: bool = True) -> set[str]:
    return {
        paper_id
        for _, paper_id in load_ready_keys(docs_root, strict=strict)
    }


def render_article(candidate: PaperCandidate, summary: PaperSummary) -> str:
    paper_id = html.escape(candidate.arxiv_id, quote=True)
    title = html.escape(candidate.title)
    one_sentence = html.escape(summary.one_sentence)
    problem = html.escape(summary.problem)
    contributions = "\n".join(f"<li>{html.escape(value)}</li>" for value in summary.contributions)
    return f'''    <article class="summary-article summary-topic-entry" id="summary-{paper_id}" data-arxiv-id="{paper_id}" data-status="ready">
<h1>[{paper_id}] {title}</h1>
<p><a href="https://arxiv.org/abs/{paper_id}" rel="nofollow">arXiv 原文</a></p>
<h2>一句话结论</h2>
<p>{one_sentence}</p>
<h2>解决的问题</h2>
<p>{problem}</p>
<h2>创新点</h2>
<ul>
{contributions}
</ul>
</article>
'''


def build_topic_document(
    path: Path,
    topic: str,
    summaries: tuple[tuple[PaperCandidate, PaperSummary], ...],
    *,
    refresh: bool = False,
) -> str:
    try:
        document = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if path.name != f"{TOPIC_SLUGS.get(topic, '')}.html":
            raise PaperSummaryError("invalid_topic", "paper topic is not publishable") from None
        title = html.escape(topic)
        slug = TOPIC_SLUGS[topic]
        payload = json.dumps({"version": 1, "topic": topic, "papers": {}}, ensure_ascii=False).replace("<", "\\u003c")
        document = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} · 论文要点</title>
  <link rel="stylesheet" href="../assets/css/site.css?v=12">
</head>
<body class="summary-page">
  <main class="summary-page-shell">
    <header class="summary-topic-header">
      <a class="summary-back" href="../index.html#{slug}">← 返回论文列表</a>
      <h1>{title} · 论文要点</h1>
    </header>
{LIST_MARKER}    </div>
  </main>
  <script type="application/json" id="summary-catalog">{payload}</script>
</body>
</html>
'''
    except OSError:
        raise PaperSummaryError("summary_page_missing", f"summary page is unavailable: {path.name}") from None
    manifest, articles = _validate(document, path, topic)
    for candidate, summary in summaries:
        if candidate.topic != topic:
            raise PaperSummaryError("topic_mismatch", f"paper topic mismatch: {candidate.arxiv_id}")
        article = render_article(candidate, summary)
        exists = candidate.arxiv_id in articles
        if exists and not refresh:
            raise PaperSummaryError("summary_exists", f"paper summary already exists: {candidate.arxiv_id}")
        if exists:
            document = document.replace(articles[candidate.arxiv_id], article, 1)
        else:
            document = document.replace(LIST_MARKER, LIST_MARKER + article, 1)
        manifest["papers"][candidate.arxiv_id] = {
            "status": "ready",
            "url": f"notes/{path.name}#summary-{candidate.arxiv_id}",
        }
    match, _ = _manifest(document, path)
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    document = document[: match.start("body")] + manifest_json + document[match.end("body") :]
    _validate(document, path, topic)
    return document


def publish_summaries(
    docs_root: str | Path,
    results: tuple[tuple[PaperCandidate, PaperSummary], ...],
    *,
    refresh: bool = False,
) -> tuple[Path, ...]:
    grouped: dict[str, list[tuple[PaperCandidate, PaperSummary]]] = {}
    for result in results:
        grouped.setdefault(result[0].topic, []).append(result)
    prepared: list[tuple[Path, str]] = []
    for topic, topic_results in grouped.items():
        path = notes_path(docs_root, topic)
        prepared.append(
            (path, build_topic_document(path, topic, tuple(topic_results), refresh=refresh))
        )
    for path, document in prepared:
        atomic_write_text(path, document)
    return tuple(path for path, _ in prepared)
