#!/usr/bin/env python3
"""Prepare a read-only evidence packet for one language-sheet agent review.

The caller (a frontier lead agent) owns LLM/subagent execution.  This script
owns only workbook context and deterministic glossary evidence, so subagents
receive hard-rule results instead of recomputing them.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _app_pipeline as ap
from agent_review_contract import ROW_TYPES, SPECIALIST_ROLES
from workbook_inspect import CONTENT_ROW_END, CONTENT_ROW_START, inspect_workbook


def _source_sheet_for(target_sheet: str, sheet_names: list[str]) -> str:
    if target_sheet in ap.GROUP_A_TARGETS:
        source = ap.GROUP_A_SOURCE
    else:
        source = ap.GROUP_B_SOURCE
    if source not in sheet_names:
        raise ValueError(f"{target_sheet}의 source 시트가 없습니다: {source}")
    return source


def _row_key(row: int) -> str:
    # PromptBuilder's row-key policy is semantic, not an Excel formatting rule.
    # Shared with the merge so findings carry the same content-type tag.
    return ROW_TYPES.get(row, "")


async def _hard_rule_evidence(workbook: Path, source_sheet: str, target_sheet: str,
                              glossary: Path, app_root: Path,
                              activation_manifest: Path | None = None) -> list[dict[str, Any]]:
    app_root = ap.bootstrap_project(str(app_root))
    from translation_web_app.glossary_checks import GlossaryChecker
    from translation_web_app.prompt_builder import PromptBuilder
    import openpyxl

    from translation_web_app.prompt_builder import HARD_CONSTRAINT_PREAMBLE

    checker = GlossaryChecker(PromptBuilder())
    source_code = ap.DEFAULT_SHEET_LANGS[source_sheet]["code"]
    target_info = ap.DEFAULT_SHEET_LANGS[target_sheet]
    loaded = await checker.load_glossary_from_file(str(glossary), source_code)
    if not loaded.startswith("✓"):
        raise RuntimeError(f"glossary 로드 실패: {loaded}")
    if activation_manifest:
        checker.load_activation_manifest(str(activation_manifest))
    story_match = re.search(r"(?:story[_ -]?)?(\d{3})(?:\D|$)", workbook.name, re.IGNORECASE)
    story = story_match.group(1) if story_match else None

    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=False)
    try:
        source_ws, target_ws = wb[source_sheet], wb[target_sheet]
        evidence = []
        for row in range(CONTENT_ROW_START, CONTENT_ROW_END + 1):
            source = str(source_ws.cell(row, 3).value or "")
            target = str(target_ws.cell(row, 3).value or "")
            if not source and not target:
                continue
            terms = checker._get_relevant_glossary_terms(source)
            targets = [checker._get_target_val(checker.glossary[term]["targets"], target_info["code"])
                       for term in terms if term in checker.glossary]
            case_report, simple_fix = checker._analyze_sentence_case(target, target_info["lang"], [term for term in targets if term])
            issues = [
                *checker._precheck_glossary_mismatch(source, target, target_info["code"]),
                *checker._check_glossary_casing(source, target, target_info["code"]),
                *checker._check_glossary_brackets(source, target, target_info["code"], target_info["lang"], row_key=_row_key(row)),
                *checker._check_brand_concatenation(source, target, target_info["code"], target_info["lang"]),
            ]
            card = checker.resolve_constraints(source, target_info["code"], row_key=_row_key(row),
                                               story=story, cell=f"C{row}")
            validation = checker.validate_constraints(target, card)
            evidence.append({"cell": f"C{row}", "row_type": _row_key(row),
                             "hard_rule_issues": issues,
                             "sentence_case_report": case_report or None,
                             "simple_case_fix": simple_fix or None,
                             "constraint_card": card,
                             "constraint_validation": validation,
                             # Only the merge-time resolver gate reads this; it is kept out
                             # of every role slice so a specialist cannot re-derive the
                             # bracket/casing policy the card already settled.
                             "glossary_context": checker._get_glossary_context_as_dict(
                                 target_info["code"], source_text=source,
                                 skip_deactivated=True, row_key=_row_key(row))})
        return evidence, HARD_CONSTRAINT_PREAMBLE
    finally:
        wb.close()


def _cell_snapshot(workbook: Path, sheet: str) -> dict[str, str]:
    """Record the target cell values the specialists actually reviewed.

    build_review_artifacts compares this against the workbook at report time, so a
    cell edited between packet and merge is reported as drift instead of silently
    proposed over.
    """
    import openpyxl

    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=False)
    try:
        ws = wb[sheet]
        snapshot = {}
        for row in range(CONTENT_ROW_START, CONTENT_ROW_END + 1):
            value = ws.cell(row, 3).value
            if value is not None:
                snapshot[f"C{row}"] = str(value)
        return snapshot
    finally:
        wb.close()


def _packet_id(packet: dict[str, Any]) -> str:
    """Hash the evidence the review is based on, not the packet envelope.

    The hard-constraint preamble is included because it is the app's statement of
    what outranks what: if that wording changes, the specialists were working under
    different instructions and the opinions are not interchangeable.
    """
    evidence = {key: packet[key] for key in (
        "workbook_name", "target_sheet", "source_sheet", "target_sections", "source_sections",
        "deterministic_evidence", "candidate_overlay", "cell_snapshot", "hard_constraint_preamble",
    )}
    canonical = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


async def build_packet(workbook: Path, sheet: str, *, glossary: Path | None = None,
                       app_root: Path | None = None, semantic_rag_budget: int = 0,
                       activation_manifest: Path | None = None,
                       candidate_overlay: Path | None = None,
                       multi_agent: bool = False) -> dict[str, Any]:
    if semantic_rag_budget < 0:
        raise ValueError("semantic RAG budget은 0 이상이어야 합니다.")
    inspect = inspect_workbook(workbook, sheet, None, with_sections=True)
    if sheet not in inspect["sheets"] or "error" in inspect["sheets"][sheet]:
        raise ValueError(f"검수할 시트를 찾을 수 없습니다: {sheet}")
    source_sheet = _source_sheet_for(sheet, inspect["sheet_names"])
    source = inspect_workbook(workbook, source_sheet, None, with_sections=True)["sheets"][source_sheet]
    deterministic: list[dict[str, Any]] = []
    preamble = ""
    if glossary and app_root:
        deterministic, preamble = await _hard_rule_evidence(
            workbook, source_sheet, sheet, glossary, app_root,
            activation_manifest=activation_manifest)
    elif glossary or app_root:
        raise ValueError("결정론적 glossary 검사는 --glossary와 --app-root를 함께 지정해야 합니다.")
    overlay_entries: list[dict[str, Any]] = []
    if candidate_overlay:
        payload = json.loads(candidate_overlay.read_text(encoding="utf-8"))
        rows = payload.get("occurrences", payload.get("entries", []))
        if not isinstance(rows, list):
            raise ValueError("candidate overlay의 occurrences/entries는 list여야 합니다.")
        story_match = re.search(r"(?:story[_ -]?)?(\d{3})(?:\D|$)", workbook.name, re.IGNORECASE)
        story = story_match.group(1) if story_match else None
        overlay_entries = [row for row in rows if isinstance(row, dict)
                           and (story is None or str(row.get("story", "")).zfill(3) == story)
                           and str(row.get("sheet", sheet)) == sheet]
    packet = {
        "schema_version": 2,
        "kind": "agent_sheet_review_packet",
        "workbook_name": workbook.name,
        "target_sheet": sheet,
        "source_sheet": source_sheet,
        "target_sections": inspect["sheets"][sheet].get("groups", []),
        "source_sections": source.get("groups", []),
        "cell_snapshot": _cell_snapshot(workbook, sheet),
        "deterministic_evidence": deterministic,
        "deterministic_evidence_status": "available" if deterministic else "not_loaded",
        # The app's own wording for "this card outranks everything else", carried
        # verbatim so every specialist reads the same authority the app's translate
        # and audit prompts state.
        "hard_constraint_preamble": preamble,
        "candidate_overlay": overlay_entries,
        "candidate_overlay_status": "available" if candidate_overlay else "not_loaded",
        "candidate_overlay_policy": (
            "english_first_candidates; ES is reference-only; global glossary output is lexical/conflict evidence"
            if candidate_overlay else None
        ),
        # The five-role split is an escalation, not the default: the roles appear
        # only when the user approved this sheet for it, so an unsure lead cannot
        # quietly pick the five-times-slower path.
        "review_mode": "multi_agent" if multi_agent else "lead_2pass",
        "subagent_roles": list(SPECIALIST_ROLES) if multi_agent else [],
        "semantic_rag_budget": semantic_rag_budget,
        "hard_rule_policy": "resolver_card_required; proposals_must_be_validated_before_merge",
        "structure_policy": "no_deterministic_structure_checker_in_v1",
    }
    packet["packet_id"] = _packet_id(packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook")
    parser.add_argument("--sheet", required=True, help="검수할 언어 시트")
    parser.add_argument("--raw", action="store_true", help="기존 구조/section 읽기 전용 출력만 수행")
    parser.add_argument("--multi-agent", action="store_true",
                        help="5개 관점 병렬 검수(escalation)를 이 시트에 승인한다. 기본은 리드 2-pass")
    parser.add_argument("--semantic-rag-budget", type=int, default=0)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--app-root", type=Path)
    parser.add_argument("--activation-manifest", type=Path,
                        help="story/cell/term activation overlay; glossary lexical values remain authoritative")
    parser.add_argument("--candidate-overlay", type=Path,
                        help="English-first candidate ledger; injects story-specific glossary review evidence only")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        if args.raw:
            raw = inspect_workbook(Path(args.workbook).expanduser(), args.sheet, None, with_sections=True)
            print(json.dumps(raw, ensure_ascii=False, indent=2))
            return
        packet = asyncio.run(build_packet(Path(args.workbook).expanduser(), args.sheet,
                                          glossary=args.glossary, app_root=args.app_root,
                                          semantic_rag_budget=args.semantic_rag_budget,
                                          activation_manifest=args.activation_manifest,
                                          candidate_overlay=args.candidate_overlay,
                                          multi_agent=args.multi_agent))
        print(json.dumps(packet, ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        sys.exit(2)


if __name__ == "__main__":
    main()
