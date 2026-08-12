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
import agent_app_report  # noqa: E402
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
    allowed = {"sheet_reviews", "agent_runs", "deterministic_checks", "rag_usage", "human_review_queue",
               "stage_reviews", "anchoring_metrics", "cell_evidence"}
    unknown = set(context) - allowed
    if unknown:
        raise ValueError("알 수 없는 review context 필드: " + ", ".join(sorted(unknown)))
    mappings = {"rag_usage", "stage_reviews", "anchoring_metrics", "cell_evidence"}
    normalised = {key: context.get(key, {} if key in mappings else []) for key in allowed}
    for key in allowed - mappings:
        if not isinstance(normalised[key], list):
            raise ValueError(f"review context.{key}는 list여야 합니다.")
    for key in mappings:
        if not isinstance(normalised[key], dict):
            raise ValueError(f"review context.{key}는 객체여야 합니다.")
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
                # Frozen copy of what the agent proposed.  A reviewer edits `after`
                # in place, so without this the original proposal — and with it any
                # way to score accuracy — would be lost at the moment of approval.
                "proposed_after": after,
                "rule_ids": rules, "approval_status": "pending_approval",
                "reason": str(proposal.get("reason", "")),
                "origin": proposal.get("origin", "subjective_consensus"),
                "row_type": proposal.get("row_type") or row_type_for_cell(cell),
                "supporting_roles": list(proposal.get("supporting_roles", [])),
            })
        return changes, drifted
    finally:
        wb.close()


# Must be grades the app's status_for_grade actually knows: anything else is read as
# "the audit produced no verdict" and rendered as blocked.  `Pass` is not one of
# them, and sending it made a clean cell report as blocked.
_GRADE_FOR_STATUS = {
    "pass": "Excellent", "warning": "Good", "needs_revision": "Needs Revision",
    "blocked": "Needs Revision", "glossary_activation_review": "Needs Revision",
}


def _cell_results(merged: ReviewMergeResult) -> list[dict]:
    """Per-cell dicts in the shape the app's render_finding consumes.

    Every packet cell appears, reviewed or not: the app's report is a full audit of
    the sheet, and silently dropping cells the agent said nothing about would make
    "no finding" indistinguishable from "never looked at".
    """
    stage_cells = {}
    cell_review = (merged.stage_reviews or {}).get("cell_review") or {}
    for row in cell_review.get("cells", []) or []:
        stage_cells[str(row.get("cell", "")).upper()] = row
    proposals = {str(item.get("cell", "")).upper(): item for item in merged.proposals}
    queued = {str(item.get("cell", "")).upper(): item for item in merged.human_review_queue}

    evidence_by_cell = dict(merged.cell_evidence or {})
    # A cell the merge has an opinion about must appear even when the packet carried
    # no deterministic evidence for it; otherwise a proposal silently disappears from
    # the report while still sitting in the manifest.
    for cell in [*proposals, *queued, *stage_cells]:
        evidence_by_cell.setdefault(cell, {"cell": cell, "target_text": merged.cell_snapshot.get(cell, "")})

    results = []
    for cell in sorted(evidence_by_cell, key=lambda value: (len(value), value)):
        evidence = evidence_by_cell[cell]
        row = stage_cells.get(cell, {})
        status = str(row.get("status") or "")
        proposal = proposals.get(cell)
        # The applyable proposal wins over the cell agent's draft: it is what
        # survived the resolver gate, and the report must not advertise text the
        # manifest will not apply.
        suggested = str((proposal or {}).get("after") or row.get("after") or "")
        if not proposal and cell in queued:
            suggested = ""
        # The staged contract requires the app's per-category evaluation, so this
        # only falls back for the legacy 5-role path, which has no cell stage.
        evaluation = [entry for entry in row.get("evaluation", []) if isinstance(entry, dict)]
        if not evaluation and row.get("reason"):
            evaluation = [{"category": "에이전트 검수", "comment": row["reason"]}]

        # render_finding derives apply_status from constraint_validation, so it must
        # describe the *proposal* — the packet's validation describes the text as it
        # stands today. Feeding the latter made the report say not_applicable for
        # changes the manifest was shipping as pending_approval.
        queue_item = queued.get(cell)
        if proposal:
            validation = {"status": "pass"}
        elif queue_item and queue_item.get("resolver_status"):
            validation = {"status": queue_item["resolver_status"],
                          "blocked": queue_item.get("resolver_violations") or []}
        else:
            validation = evidence.get("constraint_validation") or {}
        grade = _GRADE_FOR_STATUS.get(status) or ("Needs Revision" if proposal or queue_item else "Pass")
        results.append(agent_app_report.build_cell_result(
            {**evidence, "constraint_validation": validation}, sheet=merged.sheet,
            grade=grade, evaluation=evaluation, suggested_fix=suggested,
            ai_text=str(row.get("reason") or ""),
        ))
    return results


def build_review_artifacts(workbook, merged: ReviewMergeResult, *, report_id, source_file_id,
                           deterministic_checks=None, rag_usage=None, app_root=None):
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
        "stage_reviews": merged.stage_reviews,
        "anchoring_metrics": merged.anchoring_metrics,
        "cell_evidence": merged.cell_evidence,
    }
    manifest = {
        "manifest_schema_version": 2,
        "report_id": report_id,
        "source_file_id": source_file_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packet_id": merged.packet_id,
        "sheet_status": merged.sheet_status,
        "resolver_gate": merged.resolver_gate,
        "missing_roles": merged.missing_roles,
        "anchoring": merged.anchoring,
        "row_type_counts": row_type_counts,
        "changes": changes,
        "review_context": _normalise_context(review_context),
    }
    # Rendered by the app, never here: one workbook must not produce two differently
    # shaped reports depending on which path reviewed it.
    summary = [
        f"검수 시트: `{merged.sheet}` (상태: `{merged.sheet_status}`)",
        f"수정 제안: {len(changes)}건 — 모두 `pending_approval`",
        f"사람 검토 필요: {len(queue)}건",
        f"제안문 resolver 재검증: `{merged.resolver_gate}`",
    ]
    markdown = agent_app_report.render_through_app(
        app_root,
        title=f"에이전트 검수 리포트 — {merged.sheet}",
        report_id=report_id, source_file_id=source_file_id,
        results=_cell_results(merged), summary_lines=summary,
        agent_notes=agent_app_report.notes_from_merge(merged, manifest),
        extra_front_matter={"packet_id": merged.packet_id, "sheet_status": merged.sheet_status,
                            "resolver_gate": merged.resolver_gate},
    )
    return manifest, markdown


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
