"""Label configuration and the public annotation catalog."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
import re
import unicodedata

import yaml

from papers.candidate_ledger import atomic_write_json, normalize_arxiv_id

from .models import LabelDefinition, PaperAnnotation, PaperAnnotationError


CATALOG_VERSION = 2
MAX_DETAIL_TAGS = 5
MAX_DETAIL_TAGS_PER_DIMENSION = 2
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")
ARCHIVE_TITLE = re.compile(r"^\|\*\*[^*]+\*\*\|\*\*(?P<title>.*?)\*\*\|")


def label_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")


def parse_label_definitions(raw: object) -> tuple[LabelDefinition, ...]:
    if not isinstance(raw, list) or not raw:
        raise PaperAnnotationError("invalid_label_config", "paper_labels must be a non-empty list")
    labels: list[LabelDefinition] = []
    names: set[str] = set()
    slugs: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or not {"name", "description"} <= set(item) or set(item) - {"name", "description", "aliases"}:
            raise PaperAnnotationError("invalid_label_config", "each paper label needs name and description")
        name = item["name"]
        description = item["description"]
        if (
            not isinstance(name, str)
            or not 1 <= len(name.strip()) <= 80
            or not isinstance(description, str)
            or not 1 <= len(description.strip()) <= 800
            or any(character in name for character in "<>\r\n")
            or any(character in description for character in "<>\r")
        ):
            raise PaperAnnotationError("invalid_label_config", "paper label text is invalid")
        name = " ".join(name.split())
        description = " ".join(description.split())
        slug = label_slug(name)
        if not slug or name in names or slug in slugs:
            raise PaperAnnotationError("invalid_label_config", "paper label names and slugs must be unique")
        names.add(name)
        slugs.add(slug)
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
            raise PaperAnnotationError("invalid_label_config", "topic aliases must be strings")
        labels.append(LabelDefinition(name, description, slug, aliases=tuple(aliases)))
    return tuple(labels)


def load_label_definitions(config_path: str | Path) -> tuple[LabelDefinition, ...]:
    try:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise PaperAnnotationError("invalid_label_config", "site configuration cannot be read") from None
    if not isinstance(payload, dict):
        raise PaperAnnotationError("invalid_label_config", "site configuration must be a mapping")
    return parse_label_definitions(payload.get("paper_labels"))


def load_annotation_definitions(config_path: str | Path) -> tuple[LabelDefinition, ...]:
    topics = load_label_definitions(config_path)
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    groups = payload.get("paper_tag_groups", {})
    if (not isinstance(groups, dict) or not groups
            or any(not isinstance(group, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{1,39}", group)
                   for group in groups)):
        raise PaperAnnotationError("invalid_label_config", "invalid tag groups")
    details = tuple(replace(label, group=group) for group, raw in groups.items()
                    for label in parse_label_definitions(raw))
    # A task may intentionally share a topic name (Depth Estimation/Relighting).
    if len({x.name for x in details}) != len(details) or len({x.slug for x in details}) != len(details):
        raise PaperAnnotationError("invalid_label_config", "duplicate detail tags")
    alias_owners = {}
    for topic in topics:
        for name in (topic.name, *topic.aliases):
            if name in alias_owners and alias_owners[name] != topic.name:
                raise PaperAnnotationError("invalid_label_config", "ambiguous topic alias")
            alias_owners[name] = topic.name
    return (*topics, *details)


def load_topic_tag_dimensions(
    config_path: str | Path,
    labels: tuple[LabelDefinition, ...] | None = None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Load and validate the closed, dimensioned taxonomy for every topic."""
    definitions = labels or load_annotation_definitions(config_path)
    topics = {label.name for label in definitions if label.group == "topic"}
    details = {label.name: label.group for label in definitions if label.group != "topic"}
    try:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        raw = payload.get("paper_topic_tag_dimensions") if isinstance(payload, dict) else None
    except (OSError, UnicodeError, yaml.YAMLError):
        raise PaperAnnotationError("invalid_label_config", "site configuration cannot be read") from None
    if not isinstance(raw, dict) or set(raw) != topics:
        raise PaperAnnotationError(
            "invalid_label_config",
            "paper_topic_tag_dimensions must define every configured topic exactly once",
        )
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for topic, dimensions in raw.items():
        if not isinstance(dimensions, dict) or len(dimensions) < 3:
            raise PaperAnnotationError(
                "invalid_label_config",
                f"topic taxonomy needs at least three dimensions: {topic}",
            )
        assigned: set[str] = set()
        normalized: dict[str, tuple[str, ...]] = {}
        for dimension, values in dimensions.items():
            if (not isinstance(dimension, str) or dimension not in set(details.values())
                    or not isinstance(values, list) or not values
                    or any(not isinstance(value, str) or details.get(value) != dimension for value in values)
                    or len(values) != len(set(values)) or assigned.intersection(values)):
                raise PaperAnnotationError(
                    "invalid_label_config",
                    f"invalid tag dimension for topic: {topic}/{dimension}",
                )
            normalized[dimension] = tuple(values)
            assigned.update(values)
        result[topic] = normalized
    return result


def load_topic_tag_allowlists(
    config_path: str | Path,
    labels: tuple[LabelDefinition, ...] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Flatten dimensioned taxonomies for inference and schema validation."""
    dimensions = load_topic_tag_dimensions(config_path, labels)
    return {
        topic: tuple(label for values in topic_dimensions.values() for label in values)
        for topic, topic_dimensions in dimensions.items()
    }


def annotation_labels_for_topics(
    labels: tuple[LabelDefinition, ...],
    allowlists: dict[str, tuple[str, ...]],
    topics: tuple[str, ...] | list[str] | set[str],
) -> tuple[LabelDefinition, ...]:
    """Keep all navigation topics and only detail tags allowed by active archive topics."""
    aliases = {
        alias: label.name
        for label in labels
        if label.group == "topic"
        for alias in (label.name, *label.aliases)
    }
    try:
        requested = tuple(dict.fromkeys(aliases[topic] for topic in topics))
    except (KeyError, TypeError):
        raise PaperAnnotationError("invalid_label_config", "paper topics have no configured tag allowlist") from None
    if not requested or any(topic not in allowlists for topic in requested):
        raise PaperAnnotationError("invalid_label_config", "paper topics have no configured tag allowlist")
    allowed = {name for topic in requested for name in allowlists[topic]}
    return tuple(label for label in labels if label.group == "topic" or label.name in allowed)


def filter_annotation_for_topics(
    annotation: PaperAnnotation,
    labels: tuple[LabelDefinition, ...],
    allowlists: dict[str, tuple[str, ...]],
    topics: tuple[str, ...] | list[str] | set[str],
) -> PaperAnnotation:
    """Drop legacy detail tags that are outside the paper's current topic taxonomy."""
    allowed = {
        label.name
        for label in annotation_labels_for_topics(labels, allowlists, topics)
        if label.group != "topic"
    }
    filtered = tuple(tag for tag in annotation.tags if tag in allowed)
    return replace(annotation, tags=limit_detail_tags(filtered, labels))


def migrate_annotation(value: dict, labels: tuple[LabelDefinition, ...]) -> dict:
    aliases = {alias: x.name for x in labels if x.group == "topic" for alias in (x.name, *x.aliases)}
    return {"topics": list(dict.fromkeys(aliases[t] for t in value["tags"] if t in aliases)),
            "tags": [], "paper_type": value["paper_type"], "institutions": []}


def annotation_value(value: PaperAnnotation) -> dict:
    return {"topics": list(value.topics), "tags": list(value.tags),
            "paper_type": value.paper_type, "institutions": list(value.institutions)}


def limit_detail_tags(
    tags: list[str] | tuple[str, ...],
    labels: tuple[LabelDefinition, ...],
) -> tuple[str, ...]:
    """Keep the most relevant tags within global and per-dimension limits."""
    groups = {label.name: label.group for label in labels if label.group != "topic"}
    selected: list[str] = []
    counts: dict[str, int] = {}
    for tag in tags:
        group = groups[tag]
        if counts.get(group, 0) >= MAX_DETAIL_TAGS_PER_DIMENSION:
            continue
        selected.append(tag)
        counts[group] = counts.get(group, 0) + 1
        if len(selected) == MAX_DETAIL_TAGS:
            break
    return tuple(selected)


def annotation_from_value(
    paper_id: str,
    value: object,
    labels: tuple[LabelDefinition, ...],
) -> PaperAnnotation:
    def fail():
        raise PaperAnnotationError("invalid_annotation_catalog", f"invalid annotation: {paper_id}")
    if not isinstance(value, dict) or set(value) != {"topics", "tags", "paper_type", "institutions"}:
        fail()
    for field, allowed in (("topics", {x.name for x in labels if x.group == "topic"}),
                           ("tags", {x.name for x in labels if x.group != "topic"})):
        values = value[field]
        if not isinstance(values, list) or any(not isinstance(t, str) or t not in allowed for t in values) or len(values) != len(set(values)):
            fail()
    if value["paper_type"] not in ("paper", "survey"):
        fail()
    institutions = value["institutions"]
    if not isinstance(institutions, list) or len(institutions) > 30 or any(
        not isinstance(x, str) or not 1 <= len(x.strip()) <= 400 or x != " ".join(x.split())
        or any(c in x for c in "<>\r\n") for x in institutions
    ) or len(institutions) != len(set(institutions)):
        fail()
    topics = tuple(x.name for x in labels if x.group == "topic" and x.name in value["topics"])
    tags = limit_detail_tags(value["tags"], labels)
    return PaperAnnotation(topics, tags, value["paper_type"], tuple(institutions))


def load_annotation_catalog(
    path: str | Path,
    labels: tuple[LabelDefinition, ...],
) -> dict[str, PaperAnnotation]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise PaperAnnotationError("invalid_annotation_catalog", "annotation catalog cannot be read") from None
    if not isinstance(payload, dict) or set(payload) != {"version", "papers"} or payload["version"] not in (1, CATALOG_VERSION) or not isinstance(payload["papers"], dict):
        raise PaperAnnotationError("invalid_annotation_catalog", "annotation catalog schema is invalid")
    result: dict[str, PaperAnnotation] = {}
    for paper_id, value in payload["papers"].items():
        if normalize_arxiv_id(paper_id) != paper_id or not ARXIV_ID.fullmatch(paper_id):
            raise PaperAnnotationError("invalid_annotation_catalog", f"invalid arXiv ID: {paper_id}")
        if payload["version"] == 1:
            if not isinstance(value, dict) or set(value) != {"tags", "paper_type"} or not isinstance(value["tags"], list) or any(not isinstance(t, str) for t in value["tags"]):
                raise PaperAnnotationError("invalid_annotation_catalog", f"invalid legacy annotation: {paper_id}")
            value = migrate_annotation(value, labels)
        result[paper_id] = annotation_from_value(paper_id, value, labels)
    return result


def write_annotation_catalog(path: str | Path, annotations: dict[str, PaperAnnotation]) -> None:
    atomic_write_json(
        path,
        {
            "version": CATALOG_VERSION,
            "papers": {
                paper_id: annotation_value(value)
                for paper_id, value in sorted(annotations.items())
            },
        },
    )


def archive_paper_ids(archive: dict[str, dict[str, object]]) -> set[str]:
    return {normalize_arxiv_id(paper_id) for entries in archive.values() for paper_id in entries}


def annotation_coverage(
    archive: dict[str, dict[str, object]],
    annotations: dict[str, PaperAnnotation],
) -> dict[str, int]:
    paper_ids = archive_paper_ids(archive)
    annotated = len(paper_ids & annotations.keys())
    return {"total": len(paper_ids), "annotated": annotated, "pending": len(paper_ids) - annotated}


def archive_titles(archive: dict[str, dict[str, object]]) -> dict[str, tuple[str, tuple[str, ...]]]:
    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    topic_sets: dict[str, list[str]] = {}
    titles: dict[str, str] = {}
    for topic, entries in archive.items():
        for raw_id, row in entries.items():
            paper_id = normalize_arxiv_id(raw_id)
            if not isinstance(row, str) or not (match := ARCHIVE_TITLE.match(row)):
                raise PaperAnnotationError("invalid_archive", f"invalid archive row: {paper_id}")
            title = " ".join(match.group("title").split())
            if paper_id in titles and titles[paper_id] != title:
                raise PaperAnnotationError("invalid_archive", f"conflicting archive title: {paper_id}")
            titles[paper_id] = title
            topic_sets.setdefault(paper_id, [])
            if topic not in topic_sets[paper_id]:
                topic_sets[paper_id].append(topic)
    for paper_id, title in titles.items():
        result[paper_id] = (title, tuple(topic_sets[paper_id]))
    return result
