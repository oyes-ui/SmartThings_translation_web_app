#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChatGPT for Excel/Claude for Excel/Office.js draft manifest를 안전한 edits 계약으로 정규화한다."""

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
ALLOWED_SURFACES = {"chatgpt_excel", "claude_excel", "officejs_poc"}
ALLOWED_MODES = {"preview", "draft", "delivery"}
ALLOWED_VERIFICATION = {"verified", "blocked", "fallback_delivery"}
FORBIDDEN_KEY_PARTS = ("path", "file", "secret", "token", "api_key", "apikey", "password")


def _load_json(value: str) -> Any:
    candidate = Path(value).expanduser()
    raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    return json.loads(raw)


def _reject_sensitive_fields(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            folded = str(key).lower().replace("-", "_")
            if any(part in folded for part in FORBIDDEN_KEY_PARTS):
                raise ValueError(f"민감/로컬 경로 필드는 manifest에 둘 수 없습니다: {location}.{key}")
            _reject_sensitive_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_fields(child, f"{location}[{index}]")


def _require_string(item: dict, key: str, location: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key}는 비어 있지 않은 문자열이어야 합니다.")
    return value.strip()


def normalize_manifest(raw: Any, *, require_approved: bool = False) -> dict:
    """Manifest를 검증하고, 전달 가능한 안전한 최소 구조로 정규화한다."""
    if not isinstance(raw, dict):
        raise ValueError("manifest는 JSON object여야 합니다.")
    _reject_sensitive_fields(raw)

    surface = _require_string(raw, "surface", "manifest")
    if surface not in ALLOWED_SURFACES:
        raise ValueError(f"지원하지 않는 surface: {surface}")
    mode = _require_string(raw, "mode", "manifest")
    if mode not in ALLOWED_MODES:
        raise ValueError(f"지원하지 않는 mode: {mode}")
    approval = raw.get("approval", "pending")
    if approval not in {"pending", "approved"}:
        raise ValueError("manifest.approval은 pending 또는 approved여야 합니다.")
    if require_approved and approval != "approved":
        raise ValueError("승인되지 않은 manifest는 Delivery Python 적용에 사용할 수 없습니다.")
    if require_approved and mode == "preview":
        raise ValueError("preview manifest는 Delivery Python 적용에 사용할 수 없습니다.")

    changes = raw.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("manifest.changes는 비어 있지 않은 list여야 합니다.")

    normalized_changes = []
    seen = set()
    for index, change in enumerate(changes):
        location = f"manifest.changes[{index}]"
        if not isinstance(change, dict):
            raise ValueError(f"{location}은 object여야 합니다.")
        sheet = _require_string(change, "sheet", location)
        cell = _require_string(change, "cell", location).upper()
        if not CELL_RE.fullmatch(cell):
            raise ValueError(f"{location}.cell은 A1 형식이어야 합니다: {cell}")
        if "before" not in change or "after" not in change:
            raise ValueError(f"{location}에는 before와 after가 모두 필요합니다.")
        key = (sheet, cell)
        if key in seen:
            raise ValueError(f"중복 변경 대상: {sheet}!{cell}")
        seen.add(key)
        verification = change.get("verification", "verified")
        if verification not in ALLOWED_VERIFICATION:
            raise ValueError(f"{location}.verification 값이 올바르지 않습니다: {verification}")
        if require_approved and verification != "verified":
            raise ValueError(f"{location}은 verified여야 적용할 수 있습니다.")
        normalized = {
            "sheet": sheet,
            "cell": cell,
            "before": change["before"],
            "after": change["after"],
            "verification": verification,
        }
        if isinstance(change.get("reason"), str) and change["reason"].strip():
            normalized["reason"] = change["reason"].strip()
        normalized_changes.append(normalized)

    glossary = raw.get("glossary")
    normalized_glossary = None
    if glossary is not None:
        if not isinstance(glossary, dict):
            raise ValueError("manifest.glossary는 object여야 합니다.")
        locale = _require_string(glossary, "locale", "manifest.glossary")
        normalized_glossary = {"locale": locale}
        for key in ("version", "checksum"):
            if key in glossary:
                normalized_glossary[key] = _require_string(glossary, key, "manifest.glossary")

    result = {
        "surface": surface,
        "mode": mode,
        "approval": approval,
        "changes": normalized_changes,
    }
    if normalized_glossary is not None:
        result["glossary"] = normalized_glossary
    return result


def approved_edits(raw: Any) -> list[dict]:
    """`workbook_apply_edits.py`가 사용할 승인 완료 edits만 반환한다."""
    manifest = normalize_manifest(raw, require_approved=True)
    return [
        {key: change[key] for key in ("sheet", "cell", "before", "after")}
        for change in manifest["changes"]
    ]


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatGPT for Excel/Claude for Excel live manifest 검증/정규화")
    parser.add_argument("manifest", help="manifest JSON 파일 경로 또는 inline JSON")
    parser.add_argument("--require-approved", action="store_true", help="approved draft/delivery만 허용")
    parser.add_argument("--edits-only", action="store_true", help="workbook_apply_edits용 edits list만 출력")
    parser.add_argument("--output", help="정규화 결과를 기록할 JSON 경로 (원자적 쓰기)")
    parser.add_argument("--json", action="store_true", help="JSON 출력")
    args = parser.parse_args()
    try:
        raw = _load_json(args.manifest)
        manifest = normalize_manifest(raw, require_approved=args.require_approved)
        payload = {
            "status": "ok",
            "normalized_at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest,
            "edits": approved_edits(manifest) if args.require_approved else None,
        }
        if args.output:
            _atomic_json_write(Path(args.output).expanduser(), payload)
        output = payload["edits"] if args.edits_only else payload
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        message = {"status": "error", "error": str(exc)}
        print(json.dumps(message, ensure_ascii=False) if args.json else f"❌ {message['error']}")
        sys.exit(2)


if __name__ == "__main__":
    main()
