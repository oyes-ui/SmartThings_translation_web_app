#!/usr/bin/env python3
"""Contracts for the default cell -> sheet consistency -> lead review workflow."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable

from agent_review_contract import ReviewMergeResult, apply_resolver_gate, row_type_for_cell

STAGES = ("cell_review", "sheet_consistency_review", "lead_review")
FINAL_STATUSES = {"pass", "warning", "needs_revision", "blocked", "glossary_activation_review"}
_CLEAN = {"complete", "completed", "end_turn", "stop", "no_findings"}


def packet_cells(packet: dict[str, Any]) -> list[str]:
    evidence = packet.get("deterministic_evidence") or []
    cells = [str(item.get("cell", "")).upper() for item in evidence if isinstance(item, dict) and item.get("cell")]
    return cells or sorted(str(cell).upper() for cell in (packet.get("cell_snapshot") or {}))


def validate_envelope(payload: dict[str, Any], stage: str, packet: dict[str, Any]) -> dict[str, Any]:
    if stage not in STAGES or payload.get("kind") != stage:
        raise ValueError(f"단계 산출물 kind가 {stage!r}여야 합니다.")
    if str(payload.get("packet_id", "")) != str(packet.get("packet_id", "")):
        raise ValueError(f"{stage} packet_id가 근거 패킷과 다릅니다.")
    stop = str(payload.get("stop_reason", "")).lower()
    if payload.get("status") not in {"completed", "no_findings"} or stop not in _CLEAN:
        raise ValueError(f"{stage}가 정상 완료되지 않았습니다.")
    for field in ("model", "run_id", "executed_at"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"{stage}에 실행 메타데이터 {field}가 필요합니다.")
    return payload


def _validate_evaluation(evaluation: Any, cell: str, checklist: list[str]) -> None:
    """Require the app's per-category audit, not a single free-form line.

    Rendering through the app's renderer only makes the report *look* like the
    app's. The app fills a category table per cell, and a reviewer that answers
    one summary sentence produces the same layout with none of the content — so
    the categories the app asks about are required here too.
    """
    if not isinstance(evaluation, list) or not evaluation:
        raise ValueError(f"{cell}에 evaluation 배열이 필요합니다(앱 audit 항목별 결과).")
    seen = []
    for item in evaluation:
        if not isinstance(item, dict):
            raise ValueError(f"{cell}의 evaluation 항목은 객체여야 합니다.")
        category = str(item.get("category", "")).strip()
        if not category or not str(item.get("comment", "")).strip():
            raise ValueError(f"{cell}의 evaluation 항목에는 category와 comment가 모두 필요합니다.")
        if category in seen:
            raise ValueError(f"{cell}의 evaluation에 중복 category: {category}")
        seen.append(category)
    missing = [category for category in checklist if category not in seen]
    if missing:
        raise ValueError(f"{cell}의 evaluation에 빠진 검수 항목: {', '.join(missing)}")


def validate_cell_review(payload: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    validate_envelope(payload, "cell_review", packet)
    rows = payload.get("cells")
    if not isinstance(rows, list):
        raise ValueError("cell_review.cells는 list여야 합니다.")
    expected, seen = packet_cells(packet), []
    for index, row in enumerate(rows):
        cell = str(row.get("cell", "")).upper()
        if cell in seen:
            raise ValueError(f"cell_review 중복 셀: {cell}")
        seen.append(cell)
        if row.get("status") not in FINAL_STATUSES:
            raise ValueError(f"{cell}의 status가 올바르지 않습니다.")
        if not isinstance(row.get("used_prior_cell_context"), bool):
            raise ValueError(f"{cell}에 used_prior_cell_context boolean이 필요합니다.")
        refs = row.get("prior_cell_refs")
        if not isinstance(refs, list) or not all(str(ref).upper() in expected[:index] for ref in refs):
            raise ValueError(f"{cell}의 prior_cell_refs가 올바르지 않습니다.")
        if not isinstance(row.get("prior_cell_influence"), str):
            raise ValueError(f"{cell}에 prior_cell_influence 문자열이 필요합니다.")
        _validate_evaluation(row.get("evaluation"), cell, packet.get("audit_checklist") or [])
        if row["used_prior_cell_context"] != bool(refs):
            raise ValueError(f"{cell}의 앞 셀 참조 여부와 prior_cell_refs가 일치하지 않습니다.")
        if bool(row["prior_cell_influence"].strip()) != row["used_prior_cell_context"]:
            raise ValueError(f"{cell}의 앞 셀 참조 여부와 prior_cell_influence가 일치하지 않습니다.")
        _validate_after(row, cell)
    if seen != expected:
        raise ValueError(f"cell_review는 패킷 셀을 순서대로 모두 포함해야 합니다: expected={expected}, actual={seen}")
    return payload


def validate_sheet_review(payload: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    validate_envelope(payload, "sheet_consistency_review", packet)
    issues, valid = payload.get("issues"), set(packet_cells(packet))
    if not isinstance(issues, list):
        raise ValueError("sheet_consistency_review.issues는 list여야 합니다.")
    ids = set()
    for issue in issues:
        finding_id = str(issue.get("finding_id", "")).strip()
        if not finding_id or finding_id in ids:
            raise ValueError("시트 일관성 finding_id는 비어 있지 않고 고유해야 합니다.")
        ids.add(finding_id)
        affected = [str(cell).upper() for cell in issue.get("affected_cells", [])]
        proposals = issue.get("proposals")
        if not affected or not set(affected) <= valid or not isinstance(proposals, list):
            raise ValueError(f"{finding_id}의 affected_cells/proposals가 올바르지 않습니다.")
        proposed_cells = []
        for proposal in proposals:
            cell = str(proposal.get("cell", "")).upper()
            proposed_cells.append(cell)
            _validate_after(proposal, cell, required=True)
        if set(proposed_cells) != set(affected):
            raise ValueError(f"{finding_id}는 affected_cells 각각의 전체 수정안을 제시해야 합니다.")
        if not str(issue.get("canonical_pattern", "")).strip():
            raise ValueError(f"{finding_id}에 canonical_pattern이 필요합니다.")
    return payload


def validate_lead_review(payload: dict[str, Any], packet: dict[str, Any],
                         cell_review: dict[str, Any], sheet_review: dict[str, Any]) -> dict[str, Any]:
    validate_envelope(payload, "lead_review", packet)
    decisions, expected = payload.get("decisions"), packet_cells(packet)
    if not isinstance(decisions, list):
        raise ValueError("lead_review.decisions는 list여야 합니다.")
    sheet_basis = {str(issue["finding_id"]): {str(c).upper() for c in issue["affected_cells"]}
                   for issue in sheet_review.get("issues", [])}
    seen = []
    for decision in decisions:
        cell = str(decision.get("cell", "")).upper()
        seen.append(cell)
        if decision.get("status") not in FINAL_STATUSES:
            raise ValueError(f"{cell}의 최종 status가 올바르지 않습니다.")
        refs = decision.get("basis_refs")
        if not isinstance(refs, list):
            raise ValueError(f"{cell}에 basis_refs가 필요합니다.")
        for ref in refs:
            if ref == f"cell:{cell}":
                continue
            if str(ref).startswith("sheet:") and cell in sheet_basis.get(str(ref)[6:], set()):
                continue
            raise ValueError(f"{cell}의 근거 범위를 벗어난 basis_ref: {ref}")
        _validate_after(decision, cell)
        if decision.get("status") == "needs_revision" and not refs:
            raise ValueError(f"{cell} 수정안에는 cell/sheet 근거가 필요합니다.")
    if seen != expected:
        raise ValueError("lead_review는 패킷 셀을 순서대로 모두 판정해야 합니다.")
    return payload


def _validate_after(item: dict[str, Any], cell: str, required: bool = False) -> None:
    after = item.get("after")
    needs = required or item.get("status") == "needs_revision"
    if needs and (not isinstance(after, str) or not after.strip()):
        raise ValueError(f"{cell} 수정 판정에는 셀 전체 after가 필요합니다.")
    if after is not None and not isinstance(after, str):
        raise ValueError(f"{cell}.after는 문자열 또는 null이어야 합니다.")


def gate_stage(payload: dict[str, Any], stage: str, packet: dict[str, Any],
               validate: Callable[[str, str], dict[str, Any] | None]) -> dict[str, Any]:
    """Apply the shared resolver gate to every proposed string in one stage."""
    candidates = packet.get("activation_candidates") or []
    flat: list[tuple[dict[str, Any], str, str]] = []
    if stage == "cell_review":
        validate_cell_review(payload, packet)
        flat = [(row, str(row["cell"]).upper(), f"cell:{row['cell']}")
                for row in payload["cells"] if row.get("after")]
    elif stage == "sheet_consistency_review":
        validate_sheet_review(payload, packet)
        flat = [(proposal, str(proposal["cell"]).upper(), f"sheet:{issue['finding_id']}:{proposal['cell']}")
                for issue in payload["issues"] for proposal in issue["proposals"]]
    elif stage == "lead_review":
        validate_envelope(payload, "lead_review", packet)
        flat = [(row, str(row["cell"]).upper(), str(row.get("finding_id") or f"lead-{row['cell']}"))
                for row in payload.get("decisions", []) if row.get("after")]
    else:
        raise ValueError(f"알 수 없는 stage: {stage}")

    proposals = [{"finding_id": finding, "sheet": packet.get("target_sheet", ""), "cell": cell,
                  "row_type": row_type_for_cell(cell), "after": item["after"], "rule_ids": item.get("rule_ids", [])}
                 for item, cell, finding in flat]
    merged = ReviewMergeResult(proposals, [], [], "completed", [], str(packet.get("packet_id", "")),
                               str(packet.get("target_sheet", "")), dict(packet.get("cell_snapshot", {})))
    gated = apply_resolver_gate(merged, validate, activation_candidates=candidates,
                                fail_closed=True, strict_reasons=True)
    passed = {(item["cell"], item["finding_id"]): item for item in gated.proposals}
    queued = {(item["cell"], item["finding_id"]): item for item in gated.human_review_queue}
    for item, cell, finding in flat:
        key = (cell, finding)
        if key in passed:
            verdict = passed[key]
            item["after"] = verdict["after"]
            item["resolver_status"] = "pass"
            item["resolver_repaired"] = bool(verdict.get("resolver_repaired"))
        else:
            verdict = queued[key]
            disposition = verdict["reason"]
            item["status"] = ("glossary_activation_review" if disposition == "glossary_activation_review"
                              else "blocked")
            item["resolver_disposition"] = disposition
            item["resolver_status"] = verdict.get("resolver_status", "blocked")
            item["resolver_violations"] = verdict.get("resolver_violations", [])
            item["activation_candidates"] = verdict.get("activation_candidates", [])
    payload["resolver_gate_status"] = "completed"
    payload["resolver_gate"] = gated.resolver_gate
    return payload


def merge_staged_reviews(packet: dict[str, Any], cell_review: dict[str, Any],
                         sheet_review: dict[str, Any], lead_review: dict[str, Any], validate) -> ReviewMergeResult:
    # Re-run both intermediate gates at the trust boundary.  A marker in an agent
    # JSON file is metadata, not proof that the resolver actually ran.
    cell_review = gate_stage(deepcopy(cell_review), "cell_review", packet, validate)
    sheet_review = gate_stage(deepcopy(sheet_review), "sheet_consistency_review", packet, validate)
    validate_cell_review(cell_review, packet)
    validate_sheet_review(sheet_review, packet)
    validate_lead_review(lead_review, packet, cell_review, sheet_review)
    lead_review = gate_stage(deepcopy(lead_review), "lead_review", packet, validate)
    runs = []
    for stage, payload in zip(STAGES, (cell_review, sheet_review, lead_review)):
        runs.append({"role": stage, "status": "completed", "stop_reason": payload.get("stop_reason"),
                     "model": payload.get("model"), "run_id": payload.get("run_id"),
                     "executed_at": payload.get("executed_at"),
                     "opinions": len(payload.get("cells", payload.get("issues", payload.get("decisions", []))))})
    proposals, queue = [], []
    for decision in lead_review["decisions"]:
        cell, status = str(decision["cell"]).upper(), decision["status"]
        if status == "needs_revision":
            # A model may label a cell as revised while returning its current text.
            # Never emit that as an applyable change: report builders correctly
            # reject no-op edits, and the disagreement belongs in the human queue.
            if decision.get("after") == packet.get("cell_snapshot", {}).get(cell):
                queue.append({"finding_id": decision.get("finding_id") or f"lead-{cell}",
                              "sheet": packet.get("target_sheet", ""), "cell": cell,
                              "row_type": row_type_for_cell(cell), "reason": "no_op_proposal",
                              "detail": decision.get("reason", ""),
                              "rejected_after": decision.get("after") or "",
                              "resolver_status": "not_applicable", "resolver_violations": [],
                              "activation_candidates": [], "opinions": []})
                continue
            proposals.append({"finding_id": decision.get("finding_id") or f"lead-{cell}",
                              "sheet": packet.get("target_sheet", ""), "cell": cell,
                              "row_type": row_type_for_cell(cell), "after": decision.get("after"),
                              "reason": decision.get("reason", ""), "rule_ids": list(decision.get("rule_ids", [])),
                              "supporting_roles": list(decision.get("basis_refs", [])), "origin": "lead_integrated"})
        elif status in {"warning", "blocked", "glossary_activation_review"}:
            queue.append({"finding_id": decision.get("finding_id") or f"lead-{cell}",
                          "sheet": packet.get("target_sheet", ""), "cell": cell,
                          "row_type": row_type_for_cell(cell), "reason": status,
                          "detail": decision.get("reason", ""), "rejected_after": decision.get("after") or "",
                          "resolver_status": decision.get("resolver_status", ""),
                          "resolver_violations": decision.get("resolver_violations", []),
                          "activation_candidates": decision.get("activation_candidates", []),
                          "opinions": []})
    metrics_rows = cell_review["cells"]
    referenced = [row for row in metrics_rows if row["used_prior_cell_context"]]
    metrics = {"reviewed_cells": len(metrics_rows), "referenced_cells": len(referenced),
               "reference_rate": (len(referenced) / len(metrics_rows) if metrics_rows else 0.0),
               "prior_cell_refs": sorted({str(ref).upper() for row in referenced for ref in row["prior_cell_refs"]})}
    merged = ReviewMergeResult(proposals, queue, runs, "completed", [], str(packet.get("packet_id", "")),
                               str(packet.get("target_sheet", "")), dict(packet.get("cell_snapshot", {})),
                               resolver_gate="not_run")
    merged = apply_resolver_gate(merged, validate, activation_candidates=packet.get("activation_candidates"),
                                 fail_closed=True, strict_reasons=True)
    return replace(merged, stage_reviews={"cell_review": cell_review,
                                         "sheet_consistency_review": sheet_review,
                                         "lead_review": lead_review},
                   anchoring_metrics=metrics,
                   cell_evidence={str(item.get("cell", "")).upper(): item
                                  for item in packet.get("deterministic_evidence", [])})


def incomplete_staged_result(packet: dict[str, Any], missing: list[str], errors: list[str] | None = None) -> ReviewMergeResult:
    runs = [{"role": stage, "status": "missing" if stage in missing else "incomplete", "opinions": 0}
            for stage in STAGES]
    queue = [{"finding_id": "staged-review-incomplete", "sheet": packet.get("target_sheet", ""),
              "cell": "", "row_type": "", "reason": "staged_review_incomplete",
              "detail": "; ".join(errors or missing), "opinions": []}]
    return ReviewMergeResult([], queue, runs, "incomplete", list(missing), str(packet.get("packet_id", "")),
                             str(packet.get("target_sheet", "")), dict(packet.get("cell_snapshot", {})),
                             cell_evidence={str(item.get("cell", "")).upper(): item
                                            for item in packet.get("deterministic_evidence", [])})
