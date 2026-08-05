#!/usr/bin/env python3
"""Prepare a read-only evidence packet for one language-sheet agent review.

The caller (a frontier lead agent) owns LLM/subagent execution.  This script
owns only workbook context and deterministic glossary evidence, so subagents
receive hard-rule results instead of recomputing them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _app_pipeline as ap
from agent_review_contract import SPECIALIST_ROLES
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
    return {7: "title", 8: "description", 10: "title", 11: "description", 12: "disclaimer", 13: "button",
            15: "title", 16: "description", 17: "disclaimer", 18: "button",
            20: "title", 21: "description", 22: "disclaimer", 23: "button",
            25: "title", 26: "description", 27: "disclaimer", 28: "button"}.get(row, "")


async def _hard_rule_evidence(workbook: Path, source_sheet: str, target_sheet: str,
                              glossary: Path, app_root: Path,
                              activation_manifest: Path | None = None) -> list[dict[str, Any]]:
    app_root = ap.bootstrap_project(str(app_root))
    from translation_web_app.glossary_checks import GlossaryChecker
    from translation_web_app.prompt_builder import PromptBuilder
    import openpyxl

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
            evidence.append({"cell": f"C{row}", "hard_rule_issues": issues,
                             "sentence_case_report": case_report or None,
                             "simple_case_fix": simple_fix or None,
                             "constraint_card": card,
                             "constraint_validation": validation})
        return evidence
    finally:
        wb.close()


async def build_packet(workbook: Path, sheet: str, *, glossary: Path | None = None,
                       app_root: Path | None = None, semantic_rag_budget: int = 0,
                       activation_manifest: Path | None = None) -> dict[str, Any]:
    if semantic_rag_budget < 0:
        raise ValueError("semantic RAG budget은 0 이상이어야 합니다.")
    inspect = inspect_workbook(workbook, sheet, None, with_sections=True)
    if sheet not in inspect["sheets"] or "error" in inspect["sheets"][sheet]:
        raise ValueError(f"검수할 시트를 찾을 수 없습니다: {sheet}")
    source_sheet = _source_sheet_for(sheet, inspect["sheet_names"])
    source = inspect_workbook(workbook, source_sheet, None, with_sections=True)["sheets"][source_sheet]
    deterministic: list[dict[str, Any]] = []
    if glossary and app_root:
        deterministic = await _hard_rule_evidence(workbook, source_sheet, sheet, glossary, app_root,
                                                   activation_manifest=activation_manifest)
    elif glossary or app_root:
        raise ValueError("결정론적 glossary 검사는 --glossary와 --app-root를 함께 지정해야 합니다.")
    return {
        "schema_version": 1,
        "kind": "agent_sheet_review_packet",
        "workbook_name": workbook.name,
        "target_sheet": sheet,
        "source_sheet": source_sheet,
        "target_sections": inspect["sheets"][sheet].get("groups", []),
        "source_sections": source.get("groups", []),
        "deterministic_evidence": deterministic,
        "deterministic_evidence_status": "available" if deterministic else "not_loaded",
        "subagent_roles": list(SPECIALIST_ROLES),
        "semantic_rag_budget": semantic_rag_budget,
        "hard_rule_policy": "resolver_card_required; proposals_must_be_validated_before_merge",
        "structure_policy": "no_deterministic_structure_checker_in_v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook")
    parser.add_argument("--sheet", required=True, help="검수할 언어 시트")
    parser.add_argument("--raw", action="store_true", help="기존 구조/section 읽기 전용 출력만 수행")
    parser.add_argument("--semantic-rag-budget", type=int, default=0)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--app-root", type=Path)
    parser.add_argument("--activation-manifest", type=Path,
                        help="story/cell/term activation overlay; glossary lexical values remain authoritative")
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
                                          activation_manifest=args.activation_manifest))
        print(json.dumps(packet, ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        sys.exit(2)


if __name__ == "__main__":
    main()
