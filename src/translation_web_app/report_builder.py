# -*- coding: utf-8 -*-
"""Render review reports as Markdown per docs/report_format_spec.md.

The report carries a human-readable body and a machine-readable spine at the
same time: YAML front matter for job metadata, and one YAML block per finding
carrying finding_id / status / apply_status / rule_ids. Downstream tooling
reads the YAML; people read the prose. The viewer is a presentation layer and
must never be the only thing that can parse this.

Every report-producing path in checker_service goes through here, so the cell
block shape is defined once instead of being duplicated per pipeline.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Iterable, Mapping

import yaml

REPORT_SCHEMA_VERSION = 1

# 앱 검수 등급 → report_format_spec.md 의 status enum
_GRADE_TO_STATUS = {
    "excellent": "pass",
    "good": "warning",          # 출시 가능하지만 재확인 대상
    "needs revision": "needs_revision",
}
NO_SUGGESTION = "제안 없음"


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _fence(body: str, lang: str = "text") -> str:
    """Fence a block, widening the fence if the body contains backtick runs."""
    body = _text(body)
    longest = max((len(m) for m in re.findall(r"`+", body)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{lang}\n{body}\n{ticks}"


def _yaml_block(payload: Mapping[str, Any]) -> str:
    dumped = yaml.safe_dump(
        dict(payload), allow_unicode=True, sort_keys=False, width=10**6, default_flow_style=False
    )
    return _fence(dumped.rstrip("\n"), "yaml")


def _slug(sheet_name: str) -> str:
    """'DE(독일)' -> 'DE'. Falls back to a sanitized full name."""
    head = _text(sheet_name).partition("(")[0].strip()
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "-", head).strip("-")
    return cleaned or re.sub(r"[^0-9A-Za-z_]+", "-", _text(sheet_name)).strip("-") or "sheet"


def _anchor(finding_id: str) -> str:
    return re.sub(r"[^0-9a-z]+", "-", finding_id.lower()).strip("-")


def _table_cell(value: Any) -> str:
    """Escape a value for a GFM table cell: no raw newlines, no unescaped `|`."""
    text = _text(value).strip()
    text = text.replace("|", "\\|")
    return re.sub(r"\r\n|\r|\n", "<br>", text)


# GFM alert / Obsidian callout type per section. Obsidian aliases some of
# these (important/summary -> tip's color, etc.) but the viewer plugin
# (markdown-it-obsidian-callouts) only ships icons for this exact set, so
# stick to it rather than Obsidian's full alias list.
_CALLOUT_TYPES = {
    "원문": "quote",
    "현재 번역문": "note",
    "제안 번역문": "tip",
    "대소문자 점검": "info",
    "용어집 점검": "info",
    "RAG 일관성 참고": "example",
    "역번역": "quote",
}


def _callout(title: str, body: str) -> str:
    """`> [!type] title` callout, body pre-fenced and `>`-prefixed to nest inside it.

    Both Obsidian and GitHub read this as a blockquote, so it degrades to a
    plain (unstyled but still readable) blockquote anywhere that doesn't
    recognize the `[!type]` marker -- never broken, just less colorful.
    """
    callout_type = _CALLOUT_TYPES[title]
    lines = [f"> [!{callout_type}] {title}"]
    lines.extend(f"> {line}" if line else ">" for line in body.split("\n"))
    return "\n".join(lines)


def _labeled_text(label: str, value: Any, fallback: str) -> str:
    """대소문자/용어집/역번역: 서로 독립된 자유 서술이라 각자 콜아웃으로 보존한다.

    A shared table would force these into one row shape, but each is its own
    free-text report with its own internal structure that a table cell would
    flatten. The content is still fenced inside the callout so its own
    literal Markdown-looking characters (backticks, `- ` runs) are never
    reinterpreted.
    """
    text = _text(value).strip() or fallback
    return _callout(label, _fence(text))


def _rag_case_text(case: Mapping[str, Any]) -> str:
    match_type = _text(case.get("type")).upper()
    if _text(case.get("type")).strip().lower() == "semantic" and case.get("score") is not None:
        match_type += f" ({case['score']}%)"
    header = f"{match_type} | {_text(case.get('story_id'))} | {_text(case.get('section'))}"
    return f"{header}\n- 번역: {_text(case.get('target'))}"


def _rag_callout(res: Mapping[str, Any]) -> str:
    """RAG 일관성 참고: 사례가 여러 건이면 콜아웃 하나 안에 사례별로 code fence를 따로 둔다.

    Built from rag_json (checker_service's structured per-case list), not by
    re-splitting the pre-flattened rag_text -- one case's data was never one
    string to begin with. A single merged fence would visually mash unrelated
    cases together; separate fences per case make each one scannable on its
    own, same reasoning as the AI 검수 결과 table over a single blob.
    """
    cases = None
    try:
        parsed = json.loads(_text(res.get("rag_json")).strip() or "[]")
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, list) and parsed and all(isinstance(c, dict) and "target" in c for c in parsed):
        cases = parsed

    if cases:
        body = "\n\n".join(_fence(_rag_case_text(case)) for case in cases)
    else:
        text = _text(res.get("rag_text")).strip() or "[별도 지적 사항 없음]"
        body = _fence(text)
    return _callout("RAG 일관성 참고", body)


def _details(summary: str, body: str) -> str:
    """Use Obsidian-native collapsed callouts; raw HTML is intentionally banned."""
    lines = [f"> [!example]- {summary}"]
    lines.extend(f"> {line}" if line else ">" for line in body.split("\n"))
    return "\n".join(lines)


def _ai_result_block(res: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    """AI 검수 결과: evaluation이 실제 category/comment 쌍이므로 표로 만든다.

    Built from the parsed ai_json (the model's structured output), not the
    pre-flattened ai_text string -- so category/comment pairs land in real
    table rows instead of being re-parsed out of prose. Falls back to ai_text
    for bypass/skip/error paths where there is no evaluation list at all.
    """
    evaluation = audit.get("evaluation")
    if isinstance(evaluation, list) and evaluation and all(isinstance(item, dict) for item in evaluation):
        lines = ["| 항목 | 결과 |", "| --- | --- |"]
        for item in evaluation:
            category = _table_cell(item.get("category")) or "-"
            comment = _table_cell(item.get("comment")) or "-"
            lines.append(f"| {category} | {comment} |")
        return "\n".join(lines)
    ai_text = _text(res.get("ai_text")).strip() or "[해당 없음]"
    return _fence(ai_text)


def _detail_block(res: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    blocks = [
        _labeled_text("대소문자 점검", res.get("case_section"), "[별도 지적 사항 없음]"),
        _labeled_text("용어집 점검", res.get("glossary_section"), "[별도 지적 사항 없음]"),
        _rag_callout(res),
        _labeled_text("역번역", res.get("back_translation"), "[해당 없음]"),
        f"##### AI 검수 결과\n\n{_ai_result_block(res, audit)}",
    ]
    return "\n\n".join(blocks)


def parse_audit_json(raw: Any) -> dict:
    """Best-effort parse of the stored audit payload; never raises."""
    if isinstance(raw, dict):
        return raw
    text = _text(raw).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def status_for_grade(grade: str) -> str:
    """Map an audit grade to a finding status.

    An absent or unrecognized grade means the audit produced no verdict (error
    or bypass), which is reported as ``blocked`` rather than being silently
    rounded to ``pass``.
    """
    key = _text(grade).strip().lower()
    return _GRADE_TO_STATUS.get(key, "blocked")


def build_front_matter(
    *,
    report_id: str,
    source_file_id: str,
    workflow: str = "app_review",
    status: str = "draft",
    translation_model: str | None = None,
    audit_model: str | None = None,
    rules_version: str | None = None,
    rag_sources: Iterable[str] | None = None,
    generated_at: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_id": report_id,
        "workflow": workflow,
        "status": status,
        "source_file_id": source_file_id,
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if translation_model:
        payload["translation_model"] = translation_model
    if audit_model:
        payload["audit_model"] = audit_model
    if rules_version:
        payload["rules_version"] = rules_version
    payload["rag_sources"] = list(rag_sources or [])
    if extra:
        payload.update(extra)
    return payload


def render_finding(res: Mapping[str, Any], *, label: str = "", revision: int = 1) -> str:
    """Render one cell as a `### {sheet} · {cell}` section.

    ``res`` is the per-cell dict the pipelines already build (sheet_name,
    cell_ref, source, target, case_section, glossary_section, rag_text,
    back_translation, ai_text, ai_json, rag_json).
    """
    sheet = _text(res.get("sheet_name"))
    cell = _text(res.get("cell_ref"))
    finding_id = f"{_slug(sheet)}-{cell}" if cell else _slug(sheet)

    audit = parse_audit_json(res.get("ai_json"))
    grade = _text(audit.get("grade"))
    constraint_status = _text((res.get("constraint_validation") or {}).get("status")) or "not_available"
    suggestion = _text(audit.get("suggested_fix")).strip()
    current = _text(res.get("target"))
    # A suggestion identical to the current text is not a change.
    if suggestion and suggestion == current:
        suggestion = ""

    rendered_status = "blocked" if constraint_status == "blocked" else status_for_grade(grade)
    if constraint_status == "human_review" and rendered_status == "pass":
        rendered_status = "warning"
    meta = {
        "finding_id": finding_id,
        "revision": revision,
        "status": rendered_status,
        "apply_status": "pending_approval" if suggestion and constraint_status == "pass" else "not_applicable",
        "sheet": sheet,
        "cell": cell,
        "rule_ids": [],
        "rag_evidence_ids": [],
        "constraint_status": constraint_status,
        "constraint_rule_ids": sorted({rule for term in (res.get("hard_constraint_card") or {}).get("terms", []) for rule in term.get("rule_ids", [])}),
    }
    if grade:
        meta["audit_grade"] = grade

    heading = f"### {sheet} · {cell}" if cell else f"### {sheet}"
    if label:
        heading += f" ({label.strip()})"

    parts = [
        f"{heading} {{#{_anchor(finding_id)}}}",
        "",
        _yaml_block(meta),
        "",
        _callout("원문", _fence(res.get("source"))),
        "",
        _callout("현재 번역문", _fence(current)),
        "",
        _callout("제안 번역문", _fence(suggestion or NO_SUGGESTION)),
        "",
        _details("규칙 합성 판정", _fence(json.dumps({
            "constraint_card": res.get("hard_constraint_card", {}),
            "validation": res.get("constraint_validation", {}),
        }, ensure_ascii=False, indent=2), "json")),
        "",
        "#### 검수 상세",
        "",
        _detail_block(res, audit),
        "",
        "#### 원본 검수 Payload",
        "",
        _details("펼쳐서 보기 (JSON)", _fence(_text(res.get("ai_json")) or "{}", "json")),
        "",
        "#### RAG Payload",
        "",
        _details("펼쳐서 보기 (JSON)", _fence(_text(res.get("rag_json")) or "[]", "json")),
        "",
    ]
    return "\n".join(parts)


def render_report(
    *,
    title: str,
    front_matter: Mapping[str, Any],
    findings: Iterable[str],
    summary_lines: Iterable[str] = (),
    usage_report: str = "",
) -> str:
    """Assemble a full Markdown report."""
    blocks = list(findings)
    head = yaml.safe_dump(
        dict(front_matter), allow_unicode=True, sort_keys=False, width=10**6,
        default_flow_style=False,
    ).rstrip("\n")

    parts = [f"---\n{head}\n---", "", f"# {title}", "", "## 요약", ""]
    parts.extend(f"- {line}" for line in summary_lines)
    if usage_report.strip():
        parts += ["", "### 사용량", "", _fence(usage_report.strip())]
    parts += ["", "## 셀 검수", ""]
    if blocks:
        parts.extend(blocks)
    else:
        parts += ["검수 항목이 없습니다.", ""]
    # Agent-owned sections: created empty so agents append rather than restructure.
    parts += ["## Agent Notes", "", "## Decision Log", ""]
    return "\n".join(parts).rstrip("\n") + "\n"


def render_empty_report(title: str, front_matter: Mapping[str, Any], message: str) -> str:
    return render_report(
        title=title, front_matter=front_matter, findings=[], summary_lines=[message]
    )
