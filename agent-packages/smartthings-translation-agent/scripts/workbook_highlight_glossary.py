#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workbook_highlight_glossary.py — 용어집 용어 rich text 하이라이트 적용

앱 본체의 TranslationChecker.run_highlight_only_pipeline_generator 를 호출하는
얇은 래퍼다. 하이라이트 로직은 재구현하지 않는다. 공용 부트스트랩/이벤트 헬퍼는
_app_pipeline.py 를 재사용한다.

안전 정책:
  - 원본 파일은 수정하지 않는다. 앱 파이프라인이 *_highlighted_<timestamp>.xlsx
    복사본을 만든다.
  - target 셀의 기존 텍스트 중 용어집 target term과 매칭되는 글자 조각만
    openpyxl rich text로 파란색 처리한다. 번역/수정은 하지 않는다.
  - 용어집 파일을 명시하지 않으면 app repo의 runtime/glossary/latest_glossary.csv를
    사용한다.

크레딧: 0 (LLM 호출 없음, openpyxl rich text 처리만).

사용 예:
  python scripts/workbook_highlight_glossary.py story.xlsx --sheets "BR(브라질)"
  python scripts/workbook_highlight_glossary.py story.xlsx --cell-range C7:C28 --json
  python scripts/workbook_highlight_glossary.py story.xlsx --cell-range C7:C28 --include-source-sheets
  python scripts/workbook_highlight_glossary.py story.xlsx --single-source --source-sheet "US(미국)" --sheets "BR(브라질),DE(독일)"
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


async def run_highlight(args) -> dict:
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
    if not glossary.is_file():
        raise FileNotFoundError(f"용어집 CSV를 찾을 수 없습니다: {glossary}")

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
        source_groups = ap.default_source_groups(
            selected_sheets,
            workbook_sheets,
            include_source_sheets=args.include_source_sheets,
        )
        if not source_groups:
            raise ValueError("유효한 source group이 없습니다. --single-source 또는 --sheets 값을 확인하세요.")

    # JSON 모드에서는 앱 내부 print 가 stdout(=JSON 채널)을 오염시키지 않도록
    # 초기화와 파이프라인 실행 전체를 캡처한다. yield 이벤트는 그대로 수집한다.
    init_stdout = io.StringIO()
    redirect = contextlib.redirect_stdout(init_stdout) if args.json else contextlib.nullcontext()

    events: list[dict] = []
    with redirect:
        checker = TranslationChecker(max_concurrency=max(1, args.max_concurrency), no_backtranslation=True)
        async for event in checker.run_highlight_only_pipeline_generator(
            source_file_path=str(workbook),
            cell_range=args.cell_range,
            sheet_lang_map=sheet_langs,
            glossary_file_path=str(glossary),
            selected_sheets=selected_sheets,
            source_sheet_name=source_sheet,
            source_lang=source_lang,
            source_groups=source_groups,
            include_source_sheets=args.include_source_sheets,
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

    # JSON 모드: 캡처된 앱 print 를 로그 이벤트로 흡수
    if args.json:
        for line in init_stdout.getvalue().splitlines():
            if line.strip():
                events.insert(0, {"type": "log", "message": line.strip()})

    summary = ap.event_summary(events)
    summary.update({
        "source": str(workbook),
        "glossary": str(glossary),
        "cell_range": args.cell_range,
        "selected_sheets": selected_sheets,
        "source_groups": source_groups,
        "single_source": args.single_source,
        "include_source_sheets": args.include_source_sheets,
    })
    return summary


def _report_path_for(summary: dict) -> str | None:
    """하이라이트 리포트(output_data: 불일치/괄호/대소문자 로그)를 산출본 옆에 저장한다."""
    report = summary.get("output_data")
    excel_path = summary.get("excel_path")
    if not report or not excel_path:
        return None
    out = Path(excel_path).with_suffix("")
    report_path = Path(f"{out}.highlight_report.txt")
    try:
        ap.write_text_atomic(report_path, report)
        return str(report_path)
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SmartThings 워크북의 glossary target term을 Excel rich text로 하이라이트 (크레딧 0)"
    )
    parser.add_argument("workbook", help="원본 .xlsx 경로 (수정되지 않음)")
    parser.add_argument("--cell-range", default="C7:C28", help="처리할 소스 셀 범위")
    parser.add_argument("--sheets", help="대상 시트 CSV. 예: 'BR(브라질),DE(독일)'")
    parser.add_argument("--glossary", help="용어집 CSV 경로. 기본값: runtime/glossary/latest_glossary.csv")
    parser.add_argument("--sheet-langs", help="sheet_langs JSON 파일 경로. 기본값: 앱 표준 매핑")
    parser.add_argument("--single-source", action="store_true", help="복수 source group 대신 --source-sheet 하나만 사용")
    parser.add_argument("--source-sheet", default="US(미국)", help="--single-source 사용 시 소스 시트")
    parser.add_argument(
        "--include-source-sheets",
        action="store_true",
        help="기본 그룹 모드에서 KR/US source sheet 자체도 하이라이트 대상에 포함",
    )
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--app-root", help="app repo 경로 명시")
    parser.add_argument("--json", action="store_true", help="JSON 출력")
    parser.add_argument("--verbose", action="store_true", help="progress 로그도 출력")
    args = parser.parse_args()

    try:
        res = asyncio.run(run_highlight(args))
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2))
        else:
            print(f"❌ {e}")
        sys.exit(1)

    report_path = _report_path_for(res)
    if report_path:
        res["report_path"] = report_path

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        if res["status"] == "ok":
            print("✅ 하이라이트 완료")
            print(f"   원본: {res['source']}")
            print(f"   수정본: {res.get('excel_path')}")
            if report_path:
                print(f"   리포트: {report_path} (용어집 불일치·괄호·대소문자 점검)")
        else:
            print("❌ 하이라이트 중 오류가 발생했습니다.")
            for err in res.get("errors", []):
                print(f"   - {err.get('message')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
