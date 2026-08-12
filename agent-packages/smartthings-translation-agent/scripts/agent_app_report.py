#!/usr/bin/env python3
"""Render agent review results through the app's own report renderer.

The agent package had grown a second renderer, so the same workbook produced two
differently shaped reports depending on which path reviewed it. That defeats the
reason the agent path exists: it is supposed to deliver the app's report *plus*
the sheet-level consistency the app cannot do, not a separate document.

The app's renderer was already built for this. ``render_report`` closes with

    # Agent-owned sections: created empty so agents append rather than restructure.
    parts += ["## Agent Notes", "", "## Decision Log", ""]

and ``render_finding`` already reads ``hard_constraint_card`` and
``constraint_validation`` and emits ``apply_status: pending_approval`` — the exact
vocabulary the packet and the merge already speak. So this module builds the
per-cell dict the app's pipelines build and calls the app; it renders nothing
itself. Only the applyable ``changes[]`` manifest stays with the agent package,
because that is a different artifact from the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _app_pipeline as ap

AGENT_NOTES_HEADING = "## Agent Notes"


def cell_texts(sections: Iterable[Mapping[str, Any]] | None) -> dict[str, str]:
    """cell -> text from a packet's ``target_sections``/``source_sections``."""
    texts: dict[str, str] = {}
    for group in sections or []:
        for field in (group.get("fields") or {}).values():
            cell = str(field.get("cell") or "").upper()
            if cell:
                texts[cell] = str(field.get("text") or "")
    return texts


def audit_payload(*, grade: str = "", evaluation: list | None = None,
                  suggested_fix: str = "") -> str:
    """The app audit JSON string the renderer parses out of ``ai_json``."""
    return json.dumps({"evaluation": list(evaluation or []), "grade": grade or "",
                       "suggested_fix": suggested_fix or ""}, ensure_ascii=False)


def build_cell_result(evidence: Mapping[str, Any], *, sheet: str,
                      grade: str = "", evaluation: list | None = None,
                      suggested_fix: str = "", ai_text: str = "",
                      rag_text: str = "", rag_json: str = "[]",
                      back_translation: str = "") -> dict[str, Any]:
    """One packet cell + one agent verdict -> the app's per-cell result dict.

    ``case_section``/``glossary_section`` come straight from the packet because the
    app composed them there; recomposing the wording here is what would make the
    two reports diverge.
    """
    return {
        "sheet_name": sheet,
        "cell_ref": str(evidence.get("cell") or ""),
        "source": str(evidence.get("source_text") or ""),
        "target": str(evidence.get("target_text") or ""),
        "case_section": str(evidence.get("case_section") or "별도 지적 사항 없음."),
        "glossary_section": str(evidence.get("glossary_section") or "별도 지적 사항 없음."),
        "rag_text": rag_text or "별도 설정 없음.",
        "rag_json": rag_json or "[]",
        # The agent path runs no back-translation; saying so beats an empty section
        # that reads as "checked and found nothing".
        "back_translation": back_translation or "[에이전트 검수 경로: 역번역 미수행]",
        "ai_text": ai_text,
        "ai_json": audit_payload(grade=grade, evaluation=evaluation, suggested_fix=suggested_fix),
        "hard_constraint_card": evidence.get("constraint_card") or {},
        "constraint_validation": evidence.get("constraint_validation") or {},
    }


def append_agent_notes(report: str, notes: str) -> str:
    """Fill the section the app renderer leaves empty for exactly this."""
    body = (notes or "").strip()
    if not body:
        return report
    if AGENT_NOTES_HEADING not in report:
        return report.rstrip("\n") + f"\n\n{AGENT_NOTES_HEADING}\n\n{body}\n"
    head, _, tail = report.partition(AGENT_NOTES_HEADING)
    return f"{head}{AGENT_NOTES_HEADING}\n\n{body}\n{tail.lstrip(chr(10))}"


def render_through_app(app_root, *, title: str, report_id: str, source_file_id: str,
                       results: list[dict[str, Any]], summary_lines: Iterable[str] = (),
                       agent_notes: str = "", workflow: str = "agent_review",
                       extra_front_matter: Mapping[str, Any] | None = None) -> str:
    """Call the app's renderer; add nothing to the cell sections."""
    ap.bootstrap_project(str(app_root) if app_root else None)
    from translation_web_app.report_builder import (
        build_front_matter, render_finding, render_report,
    )

    report = render_report(
        title=title,
        front_matter=build_front_matter(
            report_id=report_id, source_file_id=source_file_id, workflow=workflow,
            extra=dict(extra_front_matter or {}),
        ),
        findings=[render_finding(result) for result in results],
        summary_lines=list(summary_lines),
    )
    return append_agent_notes(report, agent_notes)


def _opinion_line(opinion: Mapping[str, Any]) -> str:
    stance = str(opinion.get("stance") or "-")
    reason = str(opinion.get("reason") or "").strip() or "-"
    return f"  - `{opinion.get('role', '-')}` ({stance}): {reason}"


def notes_from_merge(merged, manifest: Mapping[str, Any] | None = None) -> str:
    """Everything the agent path knows that a per-cell app report has no slot for.

    The app's report answers "what is wrong with this cell". These notes answer
    "who said so, who disagreed, and what the reviewer must still decide" — which
    is the whole reason for running the agent path, and would be lost if rendering
    through the app meant rendering only what the app produces.
    """
    manifest = manifest or {}
    context = manifest.get("review_context") or {}
    lines: list[str] = []

    if getattr(merged, "sheet_status", "") and merged.sheet_status != "completed":
        lines += [f"> [!warning] 시트 검수 미완료 — 상태 `{merged.sheet_status}`", ""]
        if merged.missing_roles:
            lines += [f"- 누락/미완료 관점: {', '.join(merged.missing_roles)}",
                      "- 이 상태에서는 수정 제안을 만들지 않고 모든 후보를 사람 검토로 보냅니다.", ""]

    gate = getattr(merged, "resolver_gate", "not_run")
    lines.append(f"- 제안문 resolver 재검증: `{gate}`")
    if gate == "not_run":
        lines.append("- ⚠ 제안이 앱 resolver로 재검증되지 않았습니다.")

    if merged.proposals:
        lines += ["", "### 제안 근거", ""]
        for proposal in merged.proposals:
            lines.append(f"- `{proposal.get('cell', '-')}` [{proposal.get('finding_id', '-')}]")
            roles = proposal.get("supporting_roles") or []
            if roles:
                lines.append("  - supporting_roles:")
                lines += [f"    - {role}" for role in roles]
            independent = proposal.get("independent_supporting_roles")
            if independent is not None:
                lines.append(f"  - 독립 지지 관점: {', '.join(independent) or '없음'}")
            if proposal.get("resolver_repaired"):
                lines.append("  - resolver가 표기를 교정한 뒤 채택됨")
            for line in str(proposal.get("reason") or "").splitlines():
                if line.strip():
                    lines.append(f"  - {line.strip()}")

    queue = getattr(merged, "human_review_queue", []) or []
    if queue:
        lines += ["", "### 사람 검토 필요", ""]
        for item in queue:
            lines.append(f"- `{item.get('cell', '-')}` [{item.get('finding_id', '-')}]: "
                         f"{item.get('reason', '-')}")
            if item.get("detail"):
                lines.append(f"  - {item['detail']}")
            for opinion in item.get("opinions", []) or []:
                lines.append(_opinion_line(opinion))
            for violation in item.get("resolver_violations", []) or []:
                lines.append(f"  - resolver: `{violation.get('reason', '-')}` "
                             f"expected=`{violation.get('expected', '-')}`")
            if item.get("rejected_after"):
                lines.append(f"  - 차단된 제안: `{item['rejected_after']}`")

    counts = manifest.get("row_type_counts") or {}
    if counts:
        lines += ["", "### 콘텐츠 유형별 finding", ""]
        for row_type, tally in counts.items():
            lines.append(f"- `{row_type}`: 제안 {tally.get('changes', 0)}건 / "
                         f"검토 필요 {tally.get('queue', 0)}건")

    sheet_review = (getattr(merged, "stage_reviews", {}) or {}).get("sheet_consistency_review") or {}
    issues = sheet_review.get("issues") or []
    if issues:
        lines += ["", "### 시트 일관성 의견", ""]
        for issue in issues:
            cells = ", ".join(str(cell) for cell in issue.get("affected_cells", []))
            lines.append(f"- {issue.get('issue_id', '-')}: {issue.get('reason', '-')}")
            if issue.get("canonical_pattern"):
                lines.append(f"  - canonical: `{issue['canonical_pattern']}`")
            if cells:
                lines.append(f"  - 영향 셀: {cells}")

    metrics = getattr(merged, "anchoring_metrics", {}) or {}
    if metrics:
        lines += ["", "### 앞 셀 참조율", "",
                  f"- 검수 셀 {metrics.get('reviewed_cells', 0)}건 중 "
                  f"{metrics.get('referenced_cells', 0)}건이 앞 셀을 참조 "
                  f"(비율 {metrics.get('reference_rate', 0):.2f})"]
        if metrics.get("prior_cell_refs"):
            lines.append(f"- 참조된 셀: {', '.join(metrics['prior_cell_refs'])}")

    if getattr(merged, "anchoring", None):
        lines += ["", "### 관점 독립성 경고", ""]
        lines += [f"- {item.get('detail', '-')}" for item in merged.anchoring]
        lines.append("독립 근거가 아닐 수 있으므로 해당 관점이 지지한 제안은 사람이 별도 확인해야 합니다.")

    usage = context.get("rag_usage") or {}
    if usage:
        lines += ["", "### RAG 사용량", "",
                  f"- semantic: {usage.get('semantic_used', 0)} / {usage.get('semantic_budget', 0)}"]
        lines += [f"- {key}: {value}" for key, value in usage.items()
                  if key not in {"semantic_used", "semantic_budget"}]

    runs = context.get("agent_runs") or getattr(merged, "agent_runs", []) or []
    if runs:
        lines += ["", "### 관점별 실행", ""]
        for run in runs:
            mark = "✅" if run.get("status") == "completed" else "❌"
            detail = ", ".join(f"{key}={run[key]}" for key in ("model", "stop_reason", "error")
                               if run.get(key))
            lines.append(f"- {mark} `{run.get('role', '-')}`: {run.get('status', '-')} "
                         f"(의견 {run.get('opinions', 0)}건)" + (f" — {detail}" if detail else ""))
    return "\n".join(lines)
