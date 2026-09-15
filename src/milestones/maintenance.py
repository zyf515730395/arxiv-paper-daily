"""Record and report monthly per-family milestone-model verification receipts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence
from urllib.parse import urlparse

from .catalog import find_family, iter_families, load_milestone_catalog


RECEIPT_VERSION = 1
SUCCESS_STATUSES = {"unchanged", "updated"}
CHECK_STATUSES = SUCCESS_STATUSES | {"failed"}
DEFAULT_CATALOG = Path("config/milestone_models.yaml")
DEFAULT_RECEIPTS_DIR = Path("build/reports/model-maintenance")


class MaintenanceError(ValueError):
    """Raised when maintenance evidence or a stored receipt is invalid."""


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _as_utc(value: dt.datetime | None) -> dt.datetime:
    current = value or _utc_now()
    if current.tzinfo is None:
        raise MaintenanceError("checked time must include a timezone")
    return current.astimezone(dt.timezone.utc)


def _timestamp(value: dt.datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_checked_at(value: Any) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise MaintenanceError("checked_at must be a non-empty timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise MaintenanceError("checked_at must be an ISO timestamp") from error
    if parsed.tzinfo is None:
        raise MaintenanceError("checked_at must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _validate_observed_at(value: Any, *, latest: dt.date | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MaintenanceError("observed_at must be an ISO date")
    try:
        observed = dt.date.fromisoformat(value.strip())
    except ValueError as error:
        raise MaintenanceError("observed_at must be an ISO date") from error
    if latest is not None and observed > latest:
        raise MaintenanceError("observed_at cannot be in the future")
    return observed.isoformat()


def _validate_summary(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MaintenanceError("summary must be a non-empty observation")
    return value.strip()


def _validate_evidence_urls(value: Any, *, required: bool) -> list[str]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise MaintenanceError("evidence_urls must be a list")
    urls: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise MaintenanceError("evidence_urls must contain non-empty URLs")
        url = item.strip()
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise MaintenanceError("evidence_urls must contain absolute HTTPS URLs")
        if url not in urls:
            urls.append(url)
    if required and not urls:
        raise MaintenanceError("successful checks require at least one official evidence URL")
    return urls


def _validate_attempt(value: Any, *, expected_family: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MaintenanceError("receipt attempt must be an object")
    status = value.get("status")
    if status not in CHECK_STATUSES:
        raise MaintenanceError(f"receipt status must be one of {sorted(CHECK_STATUSES)}")
    if value.get("family") != expected_family:
        raise MaintenanceError("receipt attempt family does not match its file")
    return {
        "family": expected_family,
        "status": status,
        "checked_at": _timestamp(_parse_checked_at(value.get("checked_at"))),
        "observed_at": _validate_observed_at(value.get("observed_at")),
        "summary": _validate_summary(value.get("summary")),
        "evidence_urls": _validate_evidence_urls(
            value.get("evidence_urls"), required=status in SUCCESS_STATUSES
        ),
    }


def _load_receipt(path: Path, family: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaintenanceError(f"unable to load receipt {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("version") != RECEIPT_VERSION:
        raise MaintenanceError(f"unsupported receipt: {path}")
    if payload.get("family") != family:
        raise MaintenanceError(f"receipt family does not match filename: {path}")
    last_attempt = _validate_attempt(payload.get("last_attempt"), expected_family=family)
    last_success_value = payload.get("last_success")
    last_success = None
    if last_success_value is not None:
        last_success = _validate_attempt(last_success_value, expected_family=family)
        if last_success["status"] not in SUCCESS_STATUSES:
            raise MaintenanceError("last_success must have unchanged or updated status")
    if last_attempt["status"] in SUCCESS_STATUSES and last_success != last_attempt:
        raise MaintenanceError("a successful last_attempt must also be last_success")
    return {
        "version": RECEIPT_VERSION,
        "family": family,
        "last_attempt": last_attempt,
        "last_success": last_success,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_cli_receipts_dir(
    path: str | Path, *, allowed_root: str | Path = DEFAULT_RECEIPTS_DIR
) -> Path:
    """Resolve a CLI receipt path and reject traversal outside the private root."""
    private_root = Path(allowed_root).resolve()
    candidate = Path(path).resolve()
    try:
        candidate.relative_to(private_root)
    except ValueError as error:
        raise MaintenanceError(
            f"CLI receipts must stay under the private receipt directory {private_root}"
        ) from error
    return candidate


def _family_status(
    family: dict[str, Any], receipt: dict[str, Any] | None, current: dt.datetime
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "family": family["slug"],
        "name": family["name"],
        "due": True,
        "reason": "never_checked",
        "last_attempt": None,
        "last_success": None,
    }
    if receipt is None:
        return result
    result["last_attempt"] = receipt["last_attempt"]
    result["last_success"] = receipt["last_success"]
    if receipt["last_attempt"]["status"] == "failed":
        result["reason"] = "last_attempt_failed"
        return result
    successful_on = dt.date.fromisoformat(receipt["last_success"]["observed_at"])
    if (successful_on.year, successful_on.month) == (current.year, current.month):
        result["due"] = False
        result["reason"] = "checked_this_month"
    else:
        result["reason"] = "success_from_previous_month"
    return result


def maintenance_status(
    catalog_path: str | Path = DEFAULT_CATALOG,
    receipts_dir: str | Path = DEFAULT_RECEIPTS_DIR,
    *,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Validate the catalog and return machine-readable per-family due state."""
    catalog = load_milestone_catalog(catalog_path)
    receipt_root = Path(receipts_dir)
    current = _as_utc(now)
    families: list[dict[str, Any]] = []
    for _, family in iter_families(catalog):
        receipt_path = receipt_root / f"{family['slug']}.json"
        try:
            receipt = _load_receipt(receipt_path, family["slug"])
            state = _family_status(family, receipt, current)
        except MaintenanceError as error:
            state = {
                "family": family["slug"],
                "name": family["name"],
                "due": True,
                "reason": "invalid_receipt",
                "error": str(error),
                "last_attempt": None,
                "last_success": None,
            }
        families.append(state)
    due = [item["family"] for item in families if item["due"]]
    return {
        "version": RECEIPT_VERSION,
        "generated_at": _timestamp(current),
        "due_count": len(due),
        "due_families": due,
        "families": families,
    }


def record_check(
    catalog_path: str | Path = DEFAULT_CATALOG,
    receipts_dir: str | Path = DEFAULT_RECEIPTS_DIR,
    *,
    family: str,
    status: str,
    observed_at: str,
    summary: str,
    evidence_urls: Sequence[str] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Validate and atomically record one operator-verified family check."""
    catalog = load_milestone_catalog(catalog_path)
    try:
        _, family_record = find_family(catalog, family)
    except ValueError as error:
        raise MaintenanceError(str(error)) from error
    if status not in CHECK_STATUSES:
        raise MaintenanceError(f"status must be one of {sorted(CHECK_STATUSES)}")
    current = _as_utc(now)
    attempt = {
        "family": family,
        "status": status,
        "checked_at": _timestamp(current),
        "observed_at": _validate_observed_at(observed_at, latest=current.date()),
        "summary": _validate_summary(summary),
        "evidence_urls": _validate_evidence_urls(
            list(evidence_urls) if evidence_urls is not None else [],
            required=status in SUCCESS_STATUSES,
        ),
    }
    path = Path(receipts_dir) / f"{family}.json"
    previous = _load_receipt(path, family)
    last_success = attempt if status in SUCCESS_STATUSES else (
        previous["last_success"] if previous else None
    )
    payload = {
        "version": RECEIPT_VERSION,
        "family": family,
        "last_attempt": attempt,
        "last_success": last_success,
    }
    _atomic_write_json(path, payload)
    state = _family_status(family_record, payload, current)
    return {
        "ok": True,
        "family": family,
        "name": family_record["name"],
        "status": status,
        "checked_at": attempt["checked_at"],
        "due": state["due"],
        "reason": state["reason"],
        "receipt": str(path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m milestones.maintenance",
        description="Report and record monthly model-family verification receipts.",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--receipts-dir", type=Path, default=DEFAULT_RECEIPTS_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="list due families as JSON")
    record = commands.add_parser("record", help="record one verified family check")
    record.add_argument("family", nargs="?", help="catalog family slug")
    record.add_argument("--status", choices=sorted(CHECK_STATUSES))
    record.add_argument(
        "--observed-at",
        help="UTC date when official sources were observed (YYYY-MM-DD)",
    )
    record.add_argument("--summary", help="concise evidence-backed observation or failure reason")
    record.add_argument(
        "--evidence-url",
        action="append",
        dest="evidence_urls",
        help="official HTTPS source checked; repeat for multiple sources",
    )
    record.add_argument(
        "--receipt",
        type=Path,
        help="JSON input with family, status, observed_at, summary, and evidence_urls",
    )
    return parser


def _read_input_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaintenanceError(f"unable to load input receipt {path}: {error}") from error
    if not isinstance(payload, dict):
        raise MaintenanceError("input receipt must be a JSON object")
    allowed = {"family", "status", "observed_at", "summary", "evidence_urls"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise MaintenanceError(f"input receipt contains unknown fields: {unknown}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        receipts_dir = _resolve_cli_receipts_dir(args.receipts_dir)
        if args.command == "status":
            result = maintenance_status(args.catalog, receipts_dir)
        else:
            if args.receipt:
                if any(
                    value is not None
                    for value in (
                        args.family,
                        args.status,
                        args.observed_at,
                        args.summary,
                        args.evidence_urls,
                    )
                ):
                    raise MaintenanceError("--receipt cannot be combined with inline evidence fields")
                evidence = _read_input_receipt(args.receipt)
            else:
                evidence = {
                    "family": args.family,
                    "status": args.status,
                    "observed_at": args.observed_at,
                    "summary": args.summary,
                    "evidence_urls": args.evidence_urls,
                }
            result = record_check(
                args.catalog,
                receipts_dir,
                family=evidence.get("family"),
                status=evidence.get("status"),
                observed_at=evidence.get("observed_at"),
                summary=evidence.get("summary"),
                evidence_urls=evidence.get("evidence_urls"),
            )
    except (OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
