#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workbook_audit.py — 앱 검수(inspection) 파이프라인 실행 [LLM 크레딧 소모]

앱의 TranslationChecker.run_inspection_async_generator 를 호출하는 얇은 래퍼다.
기존 번역을 (재번역 없이) 검수해 등급/지적 리포트를 만든다.

⚠️ Gemini/GPT API를 호출해 **크레딧을 소모**한다. 소량·단건 검수는
   prompt_preview.py --audit 기반 "셀프 모드"(크레딧 0)를 먼저 고려한다.
   (→ references/self-vs-pipeline.md)

안전 장치:
  - --pipeline 플래그가 없으면 실행을 거부하고 셀프 모드를 안내한다.

사용 예 (승인 + 키 있을 때만):
  python scripts/workbook_audit.py story.xlsx --pipeline --sheets "DE(독일)" --json
  python scripts/workbook_audit.py story.xlsx --pipeline --single-source --source-sheet "US(미국)" --sheets "DE(독일)"
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


async def run_audit(args) -> dict:
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

    # 검수는 기존 번역을 in-place로 읽는다 → target=source 파일.
    init_stdout = io.StringIO()
    redirect = contextlib.redirect_stdout(init_stdout) if args.json else contextlib.nullcontext()

    events: list[dict] = []
    with redirect:
        checker = TranslationChecker(max_concurrency=max(1, args.max_concurrency))
        async for event in checker.run_inspection_async_generator(
            source_file_path=str(workbook),
            target_file_path=str(workbook),
            cell_range=args.cell_range,
            source_lang=source_lang,
            target_lang=args.target_lang,
            target_lang_code=args.target_lang_code,
            sheet_lang_map=sheet_langs,
            selected_sheets=selected_sheets,
            glossary_file_path=glossary_path,
            source_sheet_name=source_sheet,
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
    })
    return summary


def _write_report(summary: dict, workbook: str) -> str | None:
    report = summary.get("output_data")
    if not report:
        return None
    out = Path(workbook).with_suffix("")
    report_path = Path(f"{out}.audit_report.txt")
    try:
        ap.write_text_atomic(report_path, report)
        return str(report_path)
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="앱 검수(inspection) 파이프라인 실행 [LLM 크레딧 소모]")
    parser.add_argument("workbook", help="검수할 .xlsx (기존 번역 포함)")
    parser.add_argument("--pipeline", action="store_true",
                        help="유료 LLM 파이프라인 실행 확인 플래그 (없으면 거부)")
    parser.add_argument("--cell-range", default="C7:C28")
    parser.add_argument("--sheets", help="대상 시트 CSV")
    parser.add_argument("--target-lang", default="", help="단일 모드 fallback 타겟 언어")
    parser.add_argument("--target-lang-code", help="단일 모드 fallback 타겟 코드")
    parser.add_argument("--glossary", help="용어집 CSV 경로(기본: runtime/glossary/latest_glossary.csv)")
    parser.add_argument("--sheet-langs", help="sheet_langs JSON 파일 경로")
    parser.add_argument("--single-source", action="store_true")
    parser.add_argument("--source-sheet", default="US(미국)")
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--app-root", help="app repo 경로 명시")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.pipeline:
        msg = (
            "이 작업은 LLM 크레딧을 소모합니다. 실행하려면 --pipeline 을 명시하세요.\n"
            "소량·단건이면 크레딧 0 셀프 모드를 권장합니다:\n"
            "  python scripts/prompt_preview.py --audit --text \"<원문>\" --translated \"<번역문>\" --target-lang \"<시트>\"\n"
            "자세히: references/self-vs-pipeline.md"
        )
        if args.json:
            print(json.dumps({"status": "refused", "reason": "pipeline_flag_required", "message": msg},
                             ensure_ascii=False, indent=2))
        else:
            print(f"⛔ {msg}")
        sys.exit(2)

    try:
        res = asyncio.run(run_audit(args))
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2))
        else:
            print(f"❌ {e}")
        sys.exit(1)

    report_path = _write_report(res, args.workbook)
    if report_path:
        res["report_path"] = report_path

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        if res["status"] == "ok":
            print("✅ 검수 파이프라인 완료")
            if report_path:
                print(f"   리포트: {report_path}")
        else:
            print("❌ 검수 중 오류가 발생했습니다.")
            for err in res.get("errors", []):
                print(f"   - {err.get('message')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
