#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workbook_add_target_sheet.py — 워크북에 새 언어 타겟 시트(+선택적 역번역 시트)를
추가한 사본을 생성한다 (크레딧 0).

원본은 절대 수정하지 않는다. 같은 워크북 안의 기존 시트(--template-sheet, 기본 "ES(스페인)")를
복제해 새 시트(--new-sheet, 예: "CO(콜롬비아)")를 만든다 — 스토리마다 section 개수가 달라 행
구조가 파일마다 다르므로, 반드시 같은 파일 안의 시트를 복제해야 B열 라벨 수식과 행 구조가
그 스토리와 정확히 일치한다.

멱등적이다: 대상 시트가 이미 있으면 그 시트는 건드리지 않고 성공으로 보고한다(재실행 안전).
--new-sheet, --backtranslation-sheet 각각 독립적으로 존재 여부를 확인한다.

동작:
  1. --template-sheet 를 복제해 --new-sheet 로 이름 변경, --template-sheet 바로 뒤로 이동.
     //language 셀(행 3, C열)을 --lang-code 로 설정하고, 번역 대상 텍스트 셀(C7:C28, "x"
     placeholder·헤더 행 제외)을 비운다 — 파이프라인이 실패해도 복제 원본 언어의 텍스트가
     새 시트에 잘못 남아있는 상태를 방지한다(앱은 번역 시 어차피 무조건 덮어쓰지만, 실패 시
     빈 셀이 훨씬 눈에 잘 띄는 안전장치).
  2. --backtranslation-sheet 를 지정하면, --new-sheet 바로 뒤에 같은 방식으로 시트를 하나 더
     만든다(역번역 결과 전용 — 번역 파이프라인이 --sheets 타겟으로 취급하지 않도록 별도 이름).
     이 시트의 C열에 역번역 텍스트가 채워진다(checker_service.py 의
     TranslationChecker(backtranslation_sheet=...) 참고).

사용 예:
  python workbook_add_target_sheet.py story.xlsx --new-sheet "CO(콜롬비아)" --lang-code es_CO \
      --backtranslation-sheet "CO(콜롬비아) 역번역" --out story_co_prep.xlsx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import openpyxl

# 파싱 위치 상수: workbook_inspect.py 와 동일 규약(rag_db_builder.py 기준, 무거운 의존성 없이도
# 동작해야 하므로 fallback 유지). ⚠ 값 변경 시 workbook_inspect.py 와 동일하게 유지할 것.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bootstrap as _bs
    _app_root, _src = _bs.resolve_app_root(_bs.cli_app_root_from_argv())
    if _app_root:
        _src_dir = Path(_app_root) / "src"
        if _src_dir.is_dir() and str(_src_dir) not in sys.path:
            sys.path.insert(0, str(_src_dir))
    from translation_web_app.rag_db_builder import (
        STORY_ID_CELL,
        CONTENT_ROW_START,
        CONTENT_ROW_END,
        CONTENT_COL,
    )
except Exception:
    STORY_ID_CELL = "C5"
    CONTENT_ROW_START = 7
    CONTENT_ROW_END = 28
    CONTENT_COL = 3  # C열

LANGUAGE_ROW = 3      # "//language" 라벨 행


def _clear_content(ws) -> list[str]:
    cleared = []
    for row in range(CONTENT_ROW_START, CONTENT_ROW_END + 1):
        cell = ws.cell(row=row, column=CONTENT_COL)
        val = str(cell.value).strip() if cell.value is not None else ""
        if val and val.lower() != "x":
            cell.value = None
            cleared.append(cell.coordinate)
    return cleared


def _clone_sheet(wb, template_sheet: str, title: str, insert_after: str):
    new_ws = wb.copy_worksheet(wb[template_sheet])
    new_ws.title = title
    after_idx = wb.sheetnames.index(insert_after)
    wb.move_sheet(new_ws, offset=(after_idx + 1) - wb.sheetnames.index(title))
    return new_ws


def add_target_sheet(
    workbook: Path,
    template_sheet: str,
    new_sheet: str,
    lang_code: str,
    backtranslation_sheet: str | None = None,
) -> dict:
    wb = openpyxl.load_workbook(workbook, rich_text=True)

    if template_sheet not in wb.sheetnames:
        return {"wb": wb, "status": "error", "reason": f"템플릿 시트 없음: '{template_sheet}'"}

    story_id = wb[template_sheet][STORY_ID_CELL].value
    result = {
        "wb": wb,
        "status": "ok",
        "story_id": story_id,
        "new_sheet_action": None,
        "backtranslation_sheet_action": None,
        "cleared_cells": [],
    }

    if new_sheet in wb.sheetnames:
        result["new_sheet_action"] = "skipped (이미 존재함)"
    else:
        new_ws = _clone_sheet(wb, template_sheet, new_sheet, insert_after=template_sheet)
        new_ws.cell(row=LANGUAGE_ROW, column=CONTENT_COL).value = lang_code
        result["cleared_cells"] = _clear_content(new_ws)
        result["new_sheet_action"] = "created"

    if backtranslation_sheet:
        if backtranslation_sheet in wb.sheetnames:
            result["backtranslation_sheet_action"] = "skipped (이미 존재함)"
        else:
            bt_ws = _clone_sheet(wb, template_sheet, backtranslation_sheet, insert_after=new_sheet)
            bt_ws.cell(row=LANGUAGE_ROW, column=CONTENT_COL).value = (
                f"{lang_code} 역번역 참고용 (배포 대상 아님)"
            )
            _clear_content(bt_ws)
            result["backtranslation_sheet_action"] = "created"

    result["position"] = wb.sheetnames.index(new_sheet)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workbook", help="원본 .xlsx 경로 (수정되지 않음)")
    p.add_argument("--template-sheet", default="ES(스페인)", help="복제할 기존 시트명")
    p.add_argument("--new-sheet", required=True, help="새로 만들 시트명 (예: 'CO(콜롬비아)')")
    p.add_argument("--lang-code", required=True, help="//language 셀에 넣을 값 (예: 'es_CO')")
    p.add_argument("--backtranslation-sheet",
                    help="역번역 결과를 담을 별도 시트명 (예: 'CO(콜롬비아) 역번역'). "
                         "미지정 시 생성하지 않음")
    p.add_argument("--out", required=True, help="결과를 저장할 경로")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    workbook = Path(args.workbook).expanduser()
    if not workbook.is_file():
        msg = {"status": "error", "error": f"워크북을 찾을 수 없습니다: {workbook}"}
        print(json.dumps(msg, ensure_ascii=False) if args.json else f"❌ {msg['error']}")
        sys.exit(2)

    result = add_target_sheet(
        workbook, args.template_sheet, args.new_sheet, args.lang_code,
        backtranslation_sheet=args.backtranslation_sheet,
    )

    if result["status"] == "error":
        msg = {"status": "error", "error": result["reason"]}
        print(json.dumps(msg, ensure_ascii=False) if args.json else f"❌ {result['reason']}")
        sys.exit(1)

    out_path = Path(args.out).expanduser()
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    result["wb"].save(tmp)
    os.replace(tmp, out_path)

    summary = {k: v for k, v in result.items() if k != "wb"}
    summary["output"] = str(out_path)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"✅ 완료 → {out_path}")
        print(f"   story_id={summary.get('story_id')}")
        print(f"   '{args.new_sheet}': {summary['new_sheet_action']}")
        if args.backtranslation_sheet:
            print(f"   '{args.backtranslation_sheet}': {summary['backtranslation_sheet_action']}")


if __name__ == "__main__":
    main()
