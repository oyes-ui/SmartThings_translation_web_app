#!/usr/bin/env python3
"""Build a read-only sheet review report and a pending-approval manifest.

Schema v2 accepts a review with no proposed edits.  The ``changes`` records keep
the v1 shape so workbook_review_apply.py can continue to consume approved items.

The only accepted input is a ``ReviewMergeResult``.  A lead agent therefore
cannot turn its own summary of specialist prose into ``changes[]``: there is no
parameter that takes a free-form proposal list.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_review_contract import ReviewMergeResult, row_type_for_cell  # noqa: E402

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


def _build_changes(workbook: Path, merged: ReviewMergeResult) -> tuple[list[dict], list[dict]]:
    """Turn merged proposals into changes, diverting drifted cells to the queue.

    ``before`` comes from the workbook rather than the caller, but the packet's
    snapshot still guards the gap between review time and report time: a cell
    edited in between is queued as ``source_drift`` instead of being proposed
    over with stale evidence.
    """
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=False)
    try:
        changes, drifted, seen = [], [], set()
        for index, proposal in enumerate(merged.proposals):
            if not isinstance(proposal, dict):
                raise ValueError(f"proposals[{index}]는 object여야 합니다.")
            sheet = str(proposal.get("sheet", "")).strip() or merged.sheet
            cell = str(proposal.get("cell", "")).strip().upper()
            if sheet not in wb.sheetnames or not CELL.fullmatch(cell):
                raise ValueError(f"proposals[{index}] 대상이 올바르지 않습니다.")
            if (sheet, cell) in seen:
                raise ValueError(f"중복 제안: {sheet}!{cell}")
            seen.add((sheet, cell))
            current = wb[sheet][cell].value
            before = current if proposal.get("before") is None else proposal["before"]
            after = proposal.get("after")
            if not isinstance(after, str) or not after.strip() or after == before:
                raise ValueError(f"proposals[{index}].after가 올바르지 않습니다.")
            rules = proposal.get("rule_ids", [])
            if not isinstance(rules, list) or not all(isinstance(rule, str) and rule for rule in rules):
                raise ValueError(f"proposals[{index}].rule_ids는 문자열 list여야 합니다.")
            reviewed = merged.cell_snapshot.get(cell)
            if before != current or (reviewed is not None and reviewed != str(current or "")):
                drifted.append({
                    "finding_id": proposal.get("finding_id", ""), "sheet": sheet, "cell": cell,
                    "row_type": proposal.get("row_type") or row_type_for_cell(cell),
                    "reason": "source_drift", "opinions": [],
                    "detail": "검수 시점 값과 현재 셀 값이 달라 제안을 보류했습니다.",
                })
                continue
            changes.append({
                "finding_id": proposal.get("finding_id") or f"{sheet.split('(')[0]}-{cell}",
                "sheet": sheet, "cell": cell, "before": before, "after": after,
                "rule_ids": rules, "approval_status": "pending_approval",
                "reason": str(proposal.get("reason", "")),
                "origin": proposal.get("origin", "subjective_consensus"),
                "row_type": proposal.get("row_type") or row_type_for_cell(cell),
                "supporting_roles": list(proposal.get("supporting_roles", [])),
            })
        return changes, drifted
    finally:
        wb.close()


def _change_block(change: dict) -> str:
    rules = "\n".join(f"  - {rule}" for rule in change["rule_ids"]) or "  - review-pending"
    roles = change.get("supporting_roles") or []
    roles_line = "\n".join(f"  - {role}" for role in roles) or "  - (해당 없음: 결정론적 하드룰)"
    return f"""### {change['sheet']} · {change['cell']} {{#{change['finding_id'].lower()}}}

```yaml
finding_id: {change['finding_id']}
status: needs_revision
apply_status: pending_approval
origin: {change.get('origin', 'subjective_consensus')}
row_type: {change.get('row_type') or '(미분류)'}
supporting_roles:
{roles_line}
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


def _opinion_line(opinion: dict) -> str:
    stance = opinion.get("stance", "-")
    reason = str(opinion.get("reason", "")).strip().replace("\n", " ")
    return f"  - `{opinion.get('role', '-')}` ({stance}): {reason or '사유 없음'}"


def _markdown(manifest: dict) -> str:
    context = manifest["review_context"]
    status = manifest.get("sheet_status", "completed")
    lines = [
        "---", "report_schema_version: 2", f"report_id: {manifest['report_id']}",
        "workflow: agent_sheet_review", "status: draft", f"source_file_id: {manifest['source_file_id']}",
        f"sheet_status: {status}", f"generated_at: {manifest['generated_at']}", "---", "",
        "# 번역 검수 리포트", "",
    ]
    if status == "incomplete":
        missing = ", ".join(manifest.get("missing_roles", [])) or "알 수 없음"
        lines.extend([
            "> [!warning] 시트 검수 미완료 (`incomplete`)",
            f"> 의견서가 없거나 근거 패킷과 연결되지 않은 관점: **{missing}**",
            "> 이 리포트는 수정 제안을 만들지 않습니다. 후보는 모두 사람 검토 큐로 보냈습니다.", "",
        ])
    lines.extend(["## 시트 요약", ""])
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
        mark = "❌" if run.get("status") in {"missing", "packet_mismatch"} else "✅"
        lines.append(f"- {mark} `{run.get('role', '-')}`: {run.get('status', 'completed')} "
                     f"(의견 {run.get('opinions', 0)}건)")
    if not context["agent_runs"]:
        lines.append("- 실행 기록 없음")
    if manifest.get("anchoring"):
        lines.extend(["", "## ⚠ 관점 독립성 경고", ""])
        for item in manifest["anchoring"]:
            lines.append(f"- {item.get('detail', '-')}")
        lines.append("")
        lines.append("독립 근거가 아닐 수 있으므로 해당 관점이 지지한 제안은 사람이 별도 확인해야 합니다.")
    counts = manifest.get("row_type_counts") or {}
    if counts:
        lines.extend(["", "## 콘텐츠 유형별 finding", ""])
        for row_type in ("title", "description", "disclaimer", "button", ""):
            tally = counts.get(row_type or "(미분류)")
            if tally:
                lines.append(f"- `{row_type or '(미분류)'}`: 제안 {tally['changes']}건 / "
                             f"검토 필요 {tally['queue']}건")
    lines.extend(["", "## 사람 검토 필요", ""])
    if context["human_review_queue"]:
        for item in context["human_review_queue"]:
            lines.append(f"- `{item.get('sheet', '-')}` `{item.get('cell', '-')}` "
                         f"[{item.get('finding_id', '-')}]: {item.get('reason', '-')}")
            if item.get("detail"):
                lines.append(f"  - {item['detail']}")
            for opinion in item.get("opinions", []):
                lines.append(_opinion_line(opinion))
    else:
        lines.append("- 없음")
    lines.extend(["", "## 셀 수정 제안", "", f"- 수정 제안: {len(manifest['changes'])}건", "- 적용 상태: 모두 `pending_approval`", ""])
    lines.extend(_change_block(change) for change in manifest["changes"])
    if not manifest["changes"]:
        lines.append("- 제안 없음. 원본 workbook은 변경되지 않았습니다.\n")
    return "\n".join(lines)


def build_review_artifacts(workbook, merged: ReviewMergeResult, *, report_id, source_file_id,
                           deterministic_checks=None, rag_usage=None):
    """Build the v2 report and manifest from a merge result.

    ``merged`` must be a ReviewMergeResult — passing a list raises TypeError.
    That is the enforcement point for §4-A's rule that a lead agent cannot
    author ``changes[]`` from its own summary of specialist opinions.
    """
    if not isinstance(merged, ReviewMergeResult):
        raise TypeError(
            "build_review_artifacts는 merge_subjective_opinions()의 ReviewMergeResult만 받습니다. "
            "리드 에이전트가 정리한 proposals 리스트는 입력이 될 수 없습니다."
        )
    workbook = Path(workbook)
    if not workbook.is_file():
        raise FileNotFoundError("workbook을 찾을 수 없습니다.")
    changes, drifted = _build_changes(workbook, merged)
    queue = [*merged.human_review_queue, *drifted]
    row_type_counts: dict[str, dict[str, int]] = {}
    for bucket, items in (("changes", changes), ("queue", queue)):
        for item in items:
            key = item.get("row_type") or row_type_for_cell(item.get("cell", "")) or "(미분류)"
            row_type_counts.setdefault(key, {"changes": 0, "queue": 0})[bucket] += 1
    review_context = {
        "sheet_reviews": [{"sheet": merged.sheet, "status": merged.sheet_status,
                           "missing_roles": merged.missing_roles}],
        "agent_runs": merged.agent_runs,
        "deterministic_checks": list(deterministic_checks or []),
        "rag_usage": dict(rag_usage or {}),
        "human_review_queue": queue,
    }
    manifest = {
        "manifest_schema_version": 2,
        "report_id": report_id,
        "source_file_id": source_file_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packet_id": merged.packet_id,
        "sheet_status": merged.sheet_status,
        "missing_roles": merged.missing_roles,
        "anchoring": merged.anchoring,
        "row_type_counts": row_type_counts,
        "changes": changes,
        "review_context": _normalise_context(review_context),
    }
    return manifest, _markdown(manifest)


def write_artifacts(manifest: dict, markdown: str, output_dir, report_id) -> dict[str, str]:
    """Write the report/manifest pair atomically and return their paths."""
    output = Path(output_dir).expanduser()
    report_path, manifest_path = output / f"{report_id}.md", output / f"{report_id}.manifest.json"
    atomic(report_path, markdown)
    atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {"report": str(report_path), "manifest": str(manifest_path)}


if __name__ == "__main__":  # pragma: no cover - the CLI lives in agent_sheet_merge.py
    print(json.dumps({
        "status": "error",
        "error": "이 모듈은 라이브러리입니다. 리포트 생성은 agent_sheet_merge.py를 사용하세요 "
                 "(자유 형식 proposals 입력은 §4-A에 따라 제거되었습니다).",
    }, ensure_ascii=False))
    sys.exit(2)
