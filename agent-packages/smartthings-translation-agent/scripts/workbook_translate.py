#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workbook_translate.py — 앱 번역(+검수) 파이프라인 실행 [LLM 크레딧 소모]

앱의 TranslationChecker.run_integrated_pipeline_generator 를 호출하는 얇은 래퍼다.
소스 시트를 실제로 번역하고(옵션으로 검수까지) 새 Excel을 만든다.

⚠️ 이 스크립트는 Gemini/GPT API를 호출해 **크레딧을 소모**한다. 기본 경로가 아니다.
   소량·단건 작업은 prompt_preview.py 기반 "셀프 모드"(크레딧 0)를 먼저 고려한다.
   (→ references/self-vs-pipeline.md)

안전 장치:
  - --pipeline 플래그가 없으면 실행을 거부하고 셀프 모드를 안내한다.
  - 원본은 수정하지 않는다(앱이 새 파일 생성).

사용 예 (승인 + 키 있을 때만):
  python scripts/workbook_translate.py story.xlsx --pipeline --sheets "DE(독일)" --json
  python scripts/workbook_translate.py story.xlsx --pipeline --translate-only --single-source --source-sheet "US(미국)" --sheets "DE(독일)"
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _app_pipeline as ap


async def run_translate(args) -> dict:
    app_root = ap.bootstrap_project(args.app_root)
    ap.maybe_reexec_with_app_venv(app_root)

    from dotenv import load_dotenv
    from translation_web_app.checker_service import TranslationChecker

    load_dotenv(app_root / ".env")

    workbook = Path(args.workbook).expanduser()
    if not workbook.is_file():
        raise FileNotFoundError(f"워크북을 찾을 수 없습니다: {workbook}")

    glossary = (
        Path(args.glossary).expanduser()
        if args.glossary
        else app_root / "runtime" / "glossary" / "latest_glossary.csv"
    )
    glossary_path = str(glossary) if glossary.is_file() else None

    sheet_langs = ap.load_sheet_langs(args.sheet_langs)
    selected_sheets = ap.split_sheets(args.sheets)
    workbook_sheets = ap.workbook_sheetnames(workbook)

    if args.single_source:
        source_sheet = args.source_sheet
        source_lang = sheet_langs.get(source_sheet, {}).get("lang", "English")
        source_groups = None
    else:
        source_sheet = None
        source_lang = "English"
        source_groups = ap.default_source_groups(selected_sheets, workbook_sheets)
        if not source_groups:
            raise ValueError("유효한 source group이 없습니다. --single-source 또는 --sheets 값을 확인하세요.")

    init_stdout = io.StringIO()
    redirect = contextlib.redirect_stdout(init_stdout) if args.json else contextlib.nullcontext()

    events: list[dict] = []
    with redirect:
        checker = TranslationChecker(max_concurrency=max(1, args.max_concurrency))
        async for event in checker.run_integrated_pipeline_generator(
            source_file_path=str(workbook),
            cell_range=args.cell_range,
            bx_style_on=args.bx,
            sheet_lang_map=sheet_langs,
            translation_model=args.translation_model,
            audit_model=args.audit_model,
            glossary_file_path=glossary_path,
            selected_sheets=selected_sheets,
            source_sheet_name=source_sheet,
            skip_audit=args.translate_only,
            source_lang=source_lang,
            source_groups=source_groups,
        ):
            events.append(event)
            if not args.json:
                etype = event.get("type")
                if etype == "log":
                    print(event.get("message") or event.get("log"))
                elif etype == "progress" and args.verbose:
                    print(f"{event.get('percent', 0)}% {event.get('log', '')}")
                elif etype == "error":
                    print(f"ERROR: {event.get('message')}")

    if args.json:
        for line in init_stdout.getvalue().splitlines():
            if line.strip():
                events.insert(0, {"type": "log", "message": line.strip()})

    summary = ap.event_summary(events)
    summary.update({
        "source": str(workbook),
        "glossary": glossary_path,
        "cell_range": args.cell_range,
        "selected_sheets": selected_sheets,
        "source_groups": source_groups,
        "translate_only": args.translate_only,
        "translation_model": args.translation_model,
        "audit_model": None if args.translate_only else args.audit_model,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="앱 번역(+검수) 파이프라인 실행 [LLM 크레딧 소모]"
    )
    parser.add_argument("workbook", help="원본 .xlsx 경로 (수정되지 않음)")
    parser.add_argument("--pipeline", action="store_true",
                        help="유료 LLM 파이프라인 실행 확인 플래그 (없으면 거부)")
    parser.add_argument("--translate-only", action="store_true", help="검수 생략(번역만)")
    parser.add_argument("--cell-range", default="C7:C28")
    parser.add_argument("--sheets", help="대상 시트 CSV")
    parser.add_argument("--glossary", help="용어집 CSV 경로(기본: runtime/glossary/latest_glossary.csv)")
    parser.add_argument("--sheet-langs", help="sheet_langs JSON 파일 경로")
    parser.add_argument("--single-source", action="store_true")
    parser.add_argument("--source-sheet", default="US(미국)")
    parser.add_argument("--bx", action="store_true", help="BX 스타일 적용(영어 타겟)")
    parser.add_argument("--translation-model", default="gemini-2.5-flash")
    parser.add_argument("--audit-model", default="gpt-5.2")
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--app-root", help="app repo 경로 명시")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.pipeline:
        msg = (
            "이 작업은 LLM 크레딧을 소모합니다. 실행하려면 --pipeline 을 명시하세요.\n"
            "소량·단건이면 크레딧 0 셀프 모드를 권장합니다:\n"
            "  python scripts/prompt_preview.py --text \"<원문>\" --target-lang \"<시트>\" --row-key <맥락>\n"
            "  (+ glossary_manage.py list / rag_lookup.py 로 용어·사례 확보 후 에이전트가 직접 번역)\n"
            "자세히: references/self-vs-pipeline.md"
        )
        if args.json:
            print(json.dumps({"status": "refused", "reason": "pipeline_flag_required", "message": msg},
                             ensure_ascii=False, indent=2))
        else:
            print(f"⛔ {msg}")
        sys.exit(2)

    try:
        res = asyncio.run(run_translate(args))
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2))
        else:
            print(f"❌ {e}")
        sys.exit(1)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        if res["status"] == "ok":
            print("✅ 번역 파이프라인 완료")
            print(f"   원본: {res['source']}")
            print(f"   수정본: {res.get('excel_path')}")
        else:
            print("❌ 번역 중 오류가 발생했습니다.")
            for err in res.get("errors", []):
                print(f"   - {err.get('message')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
