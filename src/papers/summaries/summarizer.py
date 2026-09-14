"""Structured, cached map-reduce through the shared local transport."""

from __future__ import annotations

import json

from papers.model_runtime import DEFAULT_MODEL_MAX_TOKENS
from shared.loopback_chat import LoopbackChatError, LoopbackChatTransport

from .cache import PaperSummaryCache, cache_key
from .models import AcquiredPaper, PaperSummary, PaperSummaryError
from .prompts import build_chunks, map_messages, reduce_messages

SUMMARY_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['one_sentence', 'problem', 'contributions'],
    'properties': {
        'one_sentence': {'type': 'string', 'minLength': 1, 'maxLength': 600, 'pattern': r'^[^"\\<>]*$'},
        'problem': {'type': 'string', 'minLength': 1, 'maxLength': 1600, 'pattern': r'^[^"\\<>]*$'},
        'contributions': {'type': 'array', 'minItems': 3, 'maxItems': 6,
                          'items': {'type': 'string', 'minLength': 1, 'maxLength': 1000, 'pattern': r'^[^"\\<>]*$'}},
    },
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate field")
        output[key] = value
    return output


def parse_summary(raw: str) -> PaperSummary:
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
        raise PaperSummaryError(
            "invalid_summary", "model output is not the required JSON object"
        ) from None
    if not isinstance(value, dict) or set(value) != {"one_sentence", "problem", "contributions"}:
        raise PaperSummaryError("invalid_summary", "model output has unexpected fields")
    if not isinstance(value["contributions"], list):
        raise PaperSummaryError(
            "invalid_summary", "model output violates the summary contract"
        )
    try:
        return PaperSummary(
            one_sentence=value["one_sentence"],
            problem=value["problem"],
            contributions=tuple(value["contributions"]),
        )
    except (TypeError, ValueError):
        raise PaperSummaryError(
            "invalid_summary", "model output violates the summary contract"
        ) from None


def _complete(
    transport: LoopbackChatTransport,
    messages: tuple[dict[str, str], ...],
    *,
    model: str,
    timeout: float,
) -> PaperSummary:
    for allowance in (DEFAULT_MODEL_MAX_TOKENS, DEFAULT_MODEL_MAX_TOKENS * 2):
        try:
            return parse_summary(transport.complete(
                (*messages, {'role': 'user', 'content': '请写完整句子。字符串内容使用中文引号或不加引号，不使用 ASCII 双引号、反斜杠或尖括号。'}), model=model, timeout=timeout,
                max_tokens=allowance, enable_thinking=False,
                json_schema=SUMMARY_SCHEMA,
            ))
        except LoopbackChatError as error:
            if error.code == 'model_output_truncated' and allowance == DEFAULT_MODEL_MAX_TOKENS:
                continue
            raise PaperSummaryError(error.code, error.message) from None


def summarize_paper(
    paper: AcquiredPaper,
    *,
    model: str,
    base_url: str,
    timeout: float,
    refresh: bool = False,
    cache: PaperSummaryCache | None = None,
) -> PaperSummary:
    summary_cache = cache or PaperSummaryCache()
    key = cache_key(paper.source_sha256, model)
    if not refresh:
        cached = summary_cache.load(key)
        if cached is not None:
            return cached
    try:
        transport = LoopbackChatTransport(base_url)
    except LoopbackChatError as error:
        raise PaperSummaryError(error.code, error.message) from None
    level = tuple(
        _complete(
            transport,
            map_messages(paper.document.title, chunk),
            model=model,
            timeout=timeout,
        )
        for chunk in build_chunks(paper.document)
    )
    while len(level) > 1:
        reduced: list[PaperSummary] = []
        for index in range(0, len(level), 2):
            batch = level[index : index + 2]
            if len(batch) == 1:
                reduced.append(batch[0])
            else:
                reduced.append(
                    _complete(
                        transport,
                        reduce_messages(paper.document.title, batch),
                        model=model,
                        timeout=timeout,
                    )
                )
        level = tuple(reduced)
    result = level[0]
    summary_cache.store(key, result)
    return result
