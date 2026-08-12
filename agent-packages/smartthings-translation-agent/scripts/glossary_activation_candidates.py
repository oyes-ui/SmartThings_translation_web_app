#!/usr/bin/env python3
"""Find glossary occurrences that read as ordinary words, for human confirmation.

The activation manifest can record that a term is active at a story/cell, and the
resolver honours ``active: false`` — but nothing ever produced those entries. The
shipped ES_CO manifest states the reason outright: "Absent blue highlight creates
no inactive decision." Only activation was recordable, so an English source saying
"a safe home" kept matching the product term ``Safe``, and the resolver demanded a
glossary target in text where a plain adjective was correct.

That gap has a measured cost. Of the 21 proposals the ES_CO consensus produced,
six were blocked as ``missing_glossary_target`` even though the reviewer accepted
them verbatim into the delivered workbook. Under a fail-closed resolver those six
corrections never reach a human at all.

This emits *candidates*, never decisions. A candidate becomes an activation entry
only after someone sets ``confirmed: true``; --emit-manifest exports exactly those.
Deactivating a term the source really did use would silently drop a required
translation, which is the more expensive mistake of the two.
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
from workbook_inspect import CONTENT_ROW_END, CONTENT_ROW_START

SCHEMA_VERSION = 1
KIND = "glossary_inactive_candidate_manifest"
BASIS = "common_noun_lowercase_in_source"
POLICY = [
    "A glossary key carrying an uppercase letter is a product/feature term.",
    "Every occurrence of that term in the cell must be lowercase to qualify — activation is keyed by (story, cell, term), so one capitalised use keeps the whole cell active.",
    "A candidate is not a decision. Only entries with confirmed: true become activation entries.",
    "Excel and the master glossary are read-only inputs.",
]


def _story_of(name: str) -> str:
    match = re.search(r"(?:story[_ -]?)?(\d{3})(?:\D|$)", name, re.IGNORECASE)
    return match.group(1) if match else ""


def find_candidates(checker, workbook: Path, source_sheet: str, target_sheet: str,
                    target_code: str) -> list[dict[str, Any]]:
    """One candidate per (story, cell, term) whose source use is entirely lowercase."""
    import openpyxl

    story = _story_of(workbook.name)
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        if source_sheet not in wb.sheetnames:
            return []
        source_ws = wb[source_sheet]
        target_ws = wb[target_sheet] if target_sheet in wb.sheetnames else None
        entries = []
        for row in range(CONTENT_ROW_START, CONTENT_ROW_END + 1):
            source = str(source_ws.cell(row, 3).value or "")
            if not source.strip() or source.strip().lower() == "x":
                continue
            surfaces: dict[str, list[str]] = {}
            for match in checker.glossary_re.finditer(source):
                surface = match.group(0)
                key = checker.glossary_map.get(surface.lower())
                if key:
                    surfaces.setdefault(key, []).append(surface)
            target_text = str(target_ws.cell(row, 3).value or "") if target_ws else ""
            for key, found in surfaces.items():
                if not any(char.isupper() for char in key):
                    continue  # an all-lowercase glossary key says nothing about usage
                if not all(surface.islower() for surface in found):
                    continue  # the feature name appears capitalised somewhere in this cell
                target_term = checker._get_target_val(checker.glossary[key]["targets"], target_code)
                entries.append({
                    "story": story, "cell": f"C{row}", "source_term": key,
                    "active": False, "activation_basis": BASIS,
                    "confirmed": False,
                    "evidence": {
                        "source_text": source,
                        "matched_surfaces": sorted(set(found)),
                        "glossary_target": target_term or "",
                        # Corroboration only: a delivered translation that never uses
                        # the target is consistent with ordinary-word usage.
                        "target_text": target_text,
                        "target_uses_glossary_term": bool(
                            target_term and target_term.lower() in target_text.lower()),
                    },
                })
        return entries
    finally:
        wb.close()


def to_activation_entries(candidates: list[dict[str, Any]], *, confirmed_only: bool = True) -> list[dict]:
    """Strip candidates down to what OccurrenceActivationManifest reads."""
    rows = [item for item in candidates if item.get("confirmed")] if confirmed_only else candidates
    return [{"story": item["story"], "cell": item["cell"], "source_term": item["source_term"],
             "active": bool(item.get("active", False)),
             "activation_basis": item.get("activation_basis", BASIS)} for item in rows]


def markdown_review(candidates: list[dict[str, Any]]) -> str:
    lines = ["# 용어집 미적용 후보 검토", "",
             f"- 후보 {len(candidates)}건", "",
             "각 항목은 원문에서 용어집 단어가 **보통명사로 쓰인 것으로 보이는** 경우다.",
             "맞으면 `confirmed: true`로 바꾼다. 틀리면 그대로 둔다 — 확인되지 않은 후보는 무시된다.", ""]
    by_term: dict[str, list[dict]] = {}
    for item in candidates:
        by_term.setdefault(item["source_term"], []).append(item)
    for term in sorted(by_term):
        rows = by_term[term]
        lines.append(f"## `{term}` — {len(rows)}건")
        lines.append("")
        for item in rows:
            evidence = item["evidence"]
            used = "예" if evidence["target_uses_glossary_term"] else "아니오"
            lines.append(f"- story_{item['story']} `{item['cell']}` — 원문 표기 "
                         f"{', '.join(repr(s) for s in evidence['matched_surfaces'])}, "
                         f"용어집 target `{evidence['glossary_target']}`, 번역문이 이 용어를 씀: {used}")
            lines.append(f"  - 원문: {evidence['source_text'][:160]}")
        lines.append("")
    return "\n".join(lines)


async def collect(workbooks: list[Path], glossary: Path, app_root: Path, *,
                  source_sheet: str, target_sheet: str) -> list[dict[str, Any]]:
    ap.bootstrap_project(str(app_root))
    from translation_web_app.glossary_checks import GlossaryChecker
    from translation_web_app.prompt_builder import PromptBuilder

    checker = GlossaryChecker(PromptBuilder())
    source_code = ap.DEFAULT_SHEET_LANGS[source_sheet]["code"]
    loaded = await checker.load_glossary_from_file(str(glossary), source_code)
    if not loaded.startswith("✓"):
        raise RuntimeError(f"glossary 로드 실패: {loaded}")
    checker._compile_glossary_re()
    target_code = ap.DEFAULT_SHEET_LANGS[target_sheet]["code"]
    candidates: list[dict[str, Any]] = []
    for workbook in workbooks:
        candidates.extend(find_candidates(checker, workbook, source_sheet, target_sheet, target_code))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbooks", nargs="+", help="검사할 workbook 경로 또는 디렉터리")
    parser.add_argument("--glossary", type=Path, required=True)
    parser.add_argument("--app-root", type=Path)
    parser.add_argument("--source-sheet", default="US(미국)")
    parser.add_argument("--target-sheet", required=True, help="예: CO(콜롬비아)")
    parser.add_argument("--output", type=Path, help="후보 JSON 출력 경로")
    parser.add_argument("--review-markdown", type=Path, help="사람 검토용 Markdown 출력 경로")
    parser.add_argument("--emit-manifest", type=Path,
                        help="confirmed: true 후보만 activation manifest 형식으로 내보낸다")
    parser.add_argument("--from-candidates", type=Path,
                        help="새로 스캔하지 않고 기존 후보 파일을 읽어 manifest만 만든다")
    args = parser.parse_args()
    try:
        if args.from_candidates:
            payload = json.loads(args.from_candidates.expanduser().read_text(encoding="utf-8"))
            candidates = payload.get("entries", [])
        else:
            paths: list[Path] = []
            for raw in args.workbooks:
                path = Path(raw).expanduser()
                paths.extend(sorted(path.glob("*.xlsx")) if path.is_dir() else [path])
            candidates = asyncio.run(collect(
                paths, args.glossary.expanduser(), args.app_root,
                source_sheet=args.source_sheet, target_sheet=args.target_sheet))
        payload = {"schema_version": SCHEMA_VERSION, "kind": KIND, "policy": POLICY,
                   "summary": {"candidates": len(candidates),
                               "confirmed": sum(1 for item in candidates if item.get("confirmed")),
                               "terms": len({item["source_term"] for item in candidates})},
                   "entries": candidates}
        if args.output:
            ap.write_text_atomic(args.output.expanduser(),
                                 json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        if args.review_markdown:
            ap.write_text_atomic(args.review_markdown.expanduser(), markdown_review(candidates))
        if args.emit_manifest:
            entries = to_activation_entries(candidates)
            ap.write_text_atomic(args.emit_manifest.expanduser(), json.dumps(
                {"schema_version": SCHEMA_VERSION, "kind": "occurrence_activation_manifest",
                 "policy": POLICY, "entries": entries}, ensure_ascii=False, indent=2) + "\n")
            payload["summary"]["emitted_manifest_entries"] = len(entries)
        print(json.dumps({"status": "ok", **payload["summary"]}, ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        sys.exit(2)


if __name__ == "__main__":
    main()
