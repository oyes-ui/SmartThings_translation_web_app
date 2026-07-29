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
    suggestion = _text(audit.get("suggested_fix")).strip()
    current = _text(res.get("target"))
    # A suggestion identical to the current text is not a change.
    if suggestion and suggestion == current:
        suggestion = ""

    meta = {
        "finding_id": finding_id,
        "revision": revision,
        "status": status_for_grade(grade),
        "apply_status": "pending_approval" if suggestion else "not_applicable",
        "sheet": sheet,
        "cell": cell,
        "rule_ids": [],
        "rag_evidence_ids": [],
    }
    if grade:
        meta["audit_grade"] = grade

    heading = f"### {sheet} · {cell}" if cell else f"### {sheet}"
    if label:
        heading += f" ({label.strip()})"

    detail_lines = [
        f"- 대소문자 점검: {_text(res.get('case_section')) or '[별도 지적 사항 없음]'}",
        f"- 용어집 점검: {_text(res.get('glossary_section')) or '[별도 지적 사항 없음]'}",
        f"- RAG 일관성 참고: {_text(res.get('rag_text')) or '[별도 지적 사항 없음]'}",
        f"- 역번역: {_text(res.get('back_translation')) or '[해당 없음]'}",
        f"- AI 검수 결과: {_text(res.get('ai_text')) or '[해당 없음]'}",
    ]

    parts = [
        f"{heading} {{#{_anchor(finding_id)}}}",
        "",
        _yaml_block(meta),
        "",
        "#### 원문",
        "",
        _fence(res.get("source")),
        "",
        "#### 현재 번역문",
        "",
        _fence(current),
        "",
        "#### 제안 번역문",
        "",
        _fence(suggestion or NO_SUGGESTION),
        "",
        "#### 검수 상세",
        "",
        "\n".join(detail_lines),
        "",
        "#### 원본 검수 Payload",
        "",
        _fence(_text(res.get("ai_json")) or "{}", "json"),
        "",
        "#### RAG Payload",
        "",
        _fence(_text(res.get("rag_json")) or "[]", "json"),
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
