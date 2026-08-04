#!/usr/bin/env python3
"""Build a read-only sheet review report and a pending-approval manifest.

Schema v2 accepts a review with no proposed edits.  The ``changes`` records keep
the v1 shape so workbook_review_apply.py can continue to consume approved items.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl

CELL = re.compile(r"^[A-Z]{1,3}[1-9][0-9]*$")


def load_json(value: str) -> Any:
    path = Path(value).expanduser()
    return json.loads(path.read_text(encoding="utf-8") if path.is_file() else value)


def atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _normalise_context(context: dict[str, Any] | None) -> dict[str, Any]:
    context = context or {}
    if not isinstance(context, dict):
        raise ValueError("review context는 객체여야 합니다.")
    allowed = {"sheet_reviews", "agent_runs", "deterministic_checks", "rag_usage", "human_review_queue"}
    unknown = set(context) - allowed
    if unknown:
        raise ValueError("알 수 없는 review context 필드: " + ", ".join(sorted(unknown)))
    normalised = {key: context.get(key, [] if key != "rag_usage" else {}) for key in allowed}
    for key in allowed - {"rag_usage"}:
        if not isinstance(normalised[key], list):
            raise ValueError(f"review context.{key}는 list여야 합니다.")
    if not isinstance(normalised["rag_usage"], dict):
        raise ValueError("review context.rag_usage는 객체여야 합니다.")
    return normalised


def _build_changes(workbook: Path, proposals: list[dict]) -> list[dict]:
    if not isinstance(proposals, list):
        raise ValueError("proposals는 list여야 합니다.")
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=False)
    try:
        changes, seen = [], set()
        for index, proposal in enumerate(proposals):
            if not isinstance(proposal, dict):
                raise ValueError(f"proposals[{index}]는 object여야 합니다.")
            sheet = str(proposal.get("sheet", "")).strip()
            cell = str(proposal.get("cell", "")).strip().upper()
            if sheet not in wb.sheetnames or not CELL.fullmatch(cell):
                raise ValueError(f"proposals[{index}] 대상이 올바르지 않습니다.")
            if (sheet, cell) in seen:
                raise ValueError(f"중복 제안: {sheet}!{cell}")
            seen.add((sheet, cell))
            before, after = proposal.get("before"), proposal.get("after")
            if before != wb[sheet][cell].value:
                raise ValueError(f"proposals[{index}] before 불일치: {sheet}!{cell}")
            if not isinstance(after, str) or not after.strip() or after == before:
                raise ValueError(f"proposals[{index}].after가 올바르지 않습니다.")
            rules = proposal.get("rule_ids", [])
            if not isinstance(rules, list) or not all(isinstance(rule, str) and rule for rule in rules):
                raise ValueError(f"proposals[{index}].rule_ids는 문자열 list여야 합니다.")
            changes.append({
                "finding_id": proposal.get("finding_id") or f"{sheet.split('(')[0]}-{cell}",
                "sheet": sheet, "cell": cell, "before": before, "after": after,
                "rule_ids": rules, "approval_status": "pending_approval",
                "reason": str(proposal.get("reason", "")),
            })
        return changes
    finally:
        wb.close()


def _change_block(change: dict) -> str:
    rules = "\n".join(f"  - {rule}" for rule in change["rule_ids"]) or "  - review-pending"
    return f"""### {change['sheet']} · {change['cell']} {{#{change['finding_id'].lower()}}}

```yaml
finding_id: {change['finding_id']}
status: needs_revision
apply_status: pending_approval
rule_ids:
{rules}
```

#### 현재 번역문

```text
{change['before']}
```

#### 제안 번역문

```text
{change['after']}
```

#### 변경 이유

{change['reason'] or '- 규칙/RAG 검토 후 사람 승인 대기'}
"""


def _markdown(manifest: dict) -> str:
    context = manifest["review_context"]
    lines = [
        "---", "report_schema_version: 2", f"report_id: {manifest['report_id']}",
        "workflow: agent_sheet_review", "status: draft", f"source_file_id: {manifest['source_file_id']}",
        f"generated_at: {manifest['generated_at']}", "---", "", "# 번역 검수 리포트", "",
        "## 시트 요약", "",
    ]
    if context["sheet_reviews"]:
        for review in context["sheet_reviews"]:
            lines.append(f"- `{review.get('sheet', '-')}`: {review.get('status', 'completed')}")
    else:
        lines.append("- 시트 판정 정보 없음")
    lines.extend(["", "## 결정론적 검사", ""])
    lines.extend([f"- {item}" for item in context["deterministic_checks"]] or ["- 별도 지적 사항 없음"])
    rag = context["rag_usage"]
    lines.extend(["", "## RAG 사용량", "", f"- semantic: {rag.get('semantic_used', 0)} / {rag.get('semantic_budget', 0)}"])
    if rag.get("semantic_evidence_ids"):
        lines.append("- 근거 ID: " + ", ".join(rag["semantic_evidence_ids"]))
    lines.extend(["", "## 관점별 검수", ""])
    for run in context["agent_runs"]:
        lines.append(f"- `{run.get('role', '-')}`: {run.get('status', 'completed')}")
    if not context["agent_runs"]:
        lines.append("- 실행 기록 없음")
    lines.extend(["", "## 사람 검토 필요", ""])
    if context["human_review_queue"]:
        for item in context["human_review_queue"]:
            lines.append(f"- `{item.get('sheet', '-')}` `{item.get('cell', '-')}`: {item.get('reason', '-')}")
    else:
        lines.append("- 없음")
    lines.extend(["", "## 셀 수정 제안", "", f"- 수정 제안: {len(manifest['changes'])}건", "- 적용 상태: 모두 `pending_approval`", ""])
    lines.extend(_change_block(change) for change in manifest["changes"])
    if not manifest["changes"]:
        lines.append("- 제안 없음. 원본 workbook은 변경되지 않았습니다.\n")
    return "\n".join(lines)


def build_review_artifacts(workbook, proposals, *, report_id, source_file_id, review_context=None):
    workbook = Path(workbook)
    if not workbook.is_file():
        raise FileNotFoundError("workbook을 찾을 수 없습니다.")
    manifest = {
        "manifest_schema_version": 2,
        "report_id": report_id,
        "source_file_id": source_file_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "changes": _build_changes(workbook, proposals),
        "review_context": _normalise_context(review_context),
    }
    return manifest, _markdown(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook")
    parser.add_argument("proposals", help="JSON list 또는 파일 경로; 빈 list 허용")
    parser.add_argument("--review-context", help="schema v2 review context JSON 또는 파일 경로")
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--source-file-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        context = load_json(args.review_context) if args.review_context else None
        manifest, markdown = build_review_artifacts(
            args.workbook, load_json(args.proposals), report_id=args.report_id,
            source_file_id=args.source_file_id, review_context=context,
        )
        output = Path(args.output_dir).expanduser()
        report_path, manifest_path = output / f"{args.report_id}.md", output / f"{args.report_id}.manifest.json"
        atomic(report_path, markdown)
        atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        result = {"status": "ok", "report": str(report_path), "manifest": str(manifest_path), "findings": len(manifest["changes"])}
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else f"✅ {report_path}")
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        sys.exit(2)


if __name__ == "__main__":
    main()
