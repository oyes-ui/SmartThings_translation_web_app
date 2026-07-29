#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report_format_spec.md 승인 manifest를 안전한 workbook edits로 변환한다."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CELL_RE = re.compile(r"^[A-Z]{1,3}[1-9][0-9]*$")
APPROVAL_STATUSES = {"not_applicable", "pending_approval", "approved", "rejected", "applied"}


def _load_json(value: str) -> Any:
    candidate = Path(value).expanduser()
    raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    return json.loads(raw)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")
    return value.strip()


def normalize_report_manifest(raw: Any) -> dict:
    """공통 report manifest를 검증하고, 승인/미승인 변경을 명시적으로 분리한다."""
    if not isinstance(raw, dict):
        raise ValueError("report manifest는 JSON object여야 합니다.")
    if raw.get("manifest_schema_version") != 1:
        raise ValueError("manifest_schema_version: 1만 지원합니다.")
    report_id = _text(raw.get("report_id"), "report_id")
    source_file_id = _text(raw.get("source_file_id"), "source_file_id")
    changes = raw.get("changes")
    if not isinstance(changes, list):
        raise ValueError("changes는 list여야 합니다.")

    approved, skipped, seen = [], [], set()
    for index, change in enumerate(changes):
        prefix = f"changes[{index}]"
        if not isinstance(change, dict):
            raise ValueError(f"{prefix}는 object여야 합니다.")
        finding_id = _text(change.get("finding_id"), f"{prefix}.finding_id")
        sheet = _text(change.get("sheet"), f"{prefix}.sheet")
        cell = _text(change.get("cell"), f"{prefix}.cell").upper()
        if not CELL_RE.fullmatch(cell):
            raise ValueError(f"{prefix}.cell은 A1 형식이어야 합니다: {cell}")
        before = _text(change.get("before"), f"{prefix}.before")
        after = _text(change.get("after"), f"{prefix}.after")
        if after == "제안 없음":
            raise ValueError(f"{prefix}.after가 '제안 없음'이면 적용할 수 없습니다.")
        status = _text(change.get("approval_status"), f"{prefix}.approval_status")
        if status not in APPROVAL_STATUSES:
            raise ValueError(f"{prefix}.approval_status가 올바르지 않습니다: {status}")
        key = (sheet, cell)
        if key in seen:
            raise ValueError(f"동일 셀 변경이 중복됩니다: {sheet}!{cell}")
        seen.add(key)
        item = {
            "finding_id": finding_id, "sheet": sheet, "cell": cell,
            "before": before, "after": after, "approval_status": status,
            "rule_ids": change.get("rule_ids", []),
        }
        if not isinstance(item["rule_ids"], list) or not all(isinstance(v, str) for v in item["rule_ids"]):
            raise ValueError(f"{prefix}.rule_ids는 문자열 list여야 합니다.")
        (approved if status == "approved" else skipped).append(item)

    return {
        "manifest_schema_version": 1,
        "report_id": report_id,
        "source_file_id": source_file_id,
        "approved": approved,
        "skipped": skipped,
    }


def approved_edits(raw: Any) -> list[dict]:
    normalized = normalize_report_manifest(raw)
    if not normalized["approved"]:
        raise ValueError("승인된(approval_status: approved) 변경이 없습니다.")
    return [
        {key: item[key] for key in ("sheet", "cell", "before", "after")}
        for item in normalized["approved"]
    ]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="공통 검수 report manifest → 승인 Excel edits 변환")
    parser.add_argument("manifest", help="report manifest JSON 파일 경로 또는 inline JSON")
    parser.add_argument("--edits-only", action="store_true", help="승인된 edits list만 출력")
    parser.add_argument("--output", help="정규화 결과 기록 경로 (원자적 쓰기)")
    parser.add_argument("--json", action="store_true", help="JSON 출력")
    args = parser.parse_args()
    try:
        normalized = normalize_report_manifest(_load_json(args.manifest))
        payload = {
            "status": "ok",
            "normalized_at": datetime.now(timezone.utc).isoformat(),
            "manifest": normalized,
            "edits": approved_edits({
                "manifest_schema_version": 1,
                "report_id": normalized["report_id"],
                "source_file_id": normalized["source_file_id"],
                "changes": normalized["approved"],
            }) if normalized["approved"] else [],
        }
        if args.output:
            _atomic_json(Path(args.output).expanduser(), payload)
        print(json.dumps(payload["edits"] if args.edits_only else payload, ensure_ascii=False, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        message = {"status": "error", "error": str(exc)}
        print(json.dumps(message, ensure_ascii=False) if args.json else f"❌ {message['error']}")
        sys.exit(2)


if __name__ == "__main__":
    main()
