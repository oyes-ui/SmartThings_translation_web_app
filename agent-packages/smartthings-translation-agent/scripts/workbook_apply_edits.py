#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workbook_apply_edits.py — 승인된 편집을 워크북에 적용 (원본 불변)

⚠ 이 스크립트는 사용자가 명시적으로 승인한 뒤에만 실행해야 한다.

안전 정책:
  - 원본 파일은 절대 수정하지 않는다. 항상 타임스탬프가 붙은 '복사본'에 기록한다.
  - 출력은 .tmp 로 먼저 쓴 뒤 os.replace() 로 교체한다 (프로젝트 atomic write 규칙).
  - cell.value 만 갱신하여 서식(폰트/색/병합)을 보존한다 (openpyxl 기본 동작).

edits JSON 형식 (파일 경로 또는 inline 문자열):
  [
    {"sheet": "JA(일본)", "row": 10, "col": "C", "before": "기존값", "after": "新しいテキスト"},
    {"sheet": "DE(독일)", "cell": "C11", "before": "Alter Text", "new_value": "Neuer Text"}
  ]
  - col 은 문자("C") 또는 숫자(3) 모두 허용. row + col 또는 cell 둘 중 하나로 지정.
  - 새 계약은 before + after 를 권장한다. 기존 new_value 형식도 하위 호환으로 허용한다.

사용 예:
  python workbook_apply_edits.py story.xlsx edits.json
  python workbook_apply_edits.py story.xlsx '[{"sheet":"JA(일본)","cell":"C10","new_value":"..."}]'
  python workbook_apply_edits.py story.xlsx edits.json --dry-run --json
  python workbook_apply_edits.py story.xlsx edits.json --json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


def _bootstrap_project():
    """app_root 의 src/ 를 sys.path 에 추가 (없어도 비치명적).

    이 스크립트는 openpyxl 만으로 동작하므로(Level 1, Excel-only) app repo 가 없어도
    예외를 던지지 않는다. bootstrap.py(sibling)가 있으면 --app-root/env/config 를 반영한다.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import bootstrap as _bs
        app_root, _src = _bs.resolve_app_root(_bs.cli_app_root_from_argv())
        if app_root:
            src_dir = Path(app_root) / "src"
            if src_dir.is_dir() and str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            return Path(app_root)
    except Exception:
        pass

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "translation_web_app").is_dir():
            src_dir = parent / "src"
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            return parent
    return None


_bootstrap_project()

import openpyxl  # noqa: E402
from openpyxl.utils import get_column_letter, column_index_from_string  # noqa: E402
from openpyxl.utils.cell import coordinate_from_string  # noqa: E402


def _load_edits(edits_arg: str) -> list[dict]:
    """edits 인자를 파일 경로 또는 inline JSON 문자열로 해석."""
    p = Path(edits_arg).expanduser()
    if p.is_file():
        raw = p.read_text(encoding="utf-8")
    else:
        raw = edits_arg
    data = json.loads(raw)
    if isinstance(data, dict) and "changes" in data:
        # ChatGPT for Excel/Office.js manifest는 승인·검증된 draft만 허용한다.
        from excel_live_manifest import approved_edits
        return approved_edits(data)
    if not isinstance(data, list):
        raise ValueError("edits 는 객체들의 리스트 또는 승인된 live manifest여야 합니다.")
    return data


def _resolve_coord(edit: dict) -> str:
    """edit 항목을 셀 좌표 문자열('C10')로 정규화."""
    if "cell" in edit and edit["cell"]:
        return str(edit["cell"]).strip().upper()
    if "row" in edit and "col" in edit:
        col = edit["col"]
        col_letter = col if isinstance(col, str) and col.isalpha() \
            else get_column_letter(int(col))
        return f"{col_letter.upper()}{int(edit['row'])}"
    raise ValueError(f"편집 항목에 'cell' 또는 'row'+'col' 이 필요합니다: {edit}")


def _new_value(edit: dict):
    if "new_value" in edit:
        return edit["new_value"]
    if "after" in edit:
        return edit["after"]
    raise ValueError(f"편집 항목에 'new_value' 또는 'after'가 필요합니다: {edit}")


def _contains_merged_cell(ws, coord: str) -> bool:
    return any(coord in merged_range for merged_range in ws.merged_cells.ranges)


def _preflight_edits(wb, edits: list[dict], *, allow_formula: bool,
                     allow_merged: bool, allow_protected: bool,
                     allow_hidden: bool) -> tuple[list[dict], list[dict]]:
    planned, errors = [], []
    for i, edit in enumerate(edits):
        try:
            sheet = edit["sheet"]
            coord = _resolve_coord(edit)
            new_value = _new_value(edit)
            if sheet not in wb.sheetnames:
                raise ValueError(f"시트 '{sheet}' 가 워크북에 없습니다.")
            ws = wb[sheet]
            cell = ws[coord]
            old_value = cell.value
            if "before" in edit and old_value != edit["before"]:
                raise ValueError(
                    f"before 불일치: 현재값={old_value!r}, 기대값={edit['before']!r}"
                )
            if ws.sheet_state != "visible" and not allow_hidden:
                raise ValueError("숨김 시트는 --allow-hidden 없이는 수정할 수 없습니다.")
            if ws.protection.sheet and not allow_protected:
                raise ValueError("보호 시트는 --allow-protected 없이는 수정할 수 없습니다.")
            if _contains_merged_cell(ws, coord) and not allow_merged:
                raise ValueError("병합 범위 셀은 --allow-merged 없이는 수정할 수 없습니다.")
            if isinstance(old_value, str) and old_value.startswith("=") and not allow_formula:
                raise ValueError("수식 셀은 --allow-formula 없이는 수정할 수 없습니다.")
            planned.append({
                "index": i,
                "sheet": sheet,
                "cell": coord,
                "old_value": old_value,
                "new_value": new_value,
                "has_rich_text_risk": isinstance(old_value, str) and bool(old_value),
            })
        except Exception as e:
            errors.append({"index": i, "edit": edit, "error": str(e)})
    return planned, errors


def _structure_snapshot(wb) -> dict:
    """편집 전후에도 변하면 안 되는 workbook 구조를 비교한다."""
    return {
        "sheetnames": list(wb.sheetnames),
        "sheets": {
            ws.title: {
                "state": ws.sheet_state,
                "merged_ranges": sorted(str(rng) for rng in ws.merged_cells.ranges),
                "protection": bool(ws.protection.sheet),
                "freeze_panes": str(ws.freeze_panes) if ws.freeze_panes else None,
            }
            for ws in wb.worksheets
        },
    }


def _value_snapshot(wb) -> dict:
    """None이 아닌 셀 값과 수식을 비교해 허용 범위 밖 변경을 찾는다."""
    snapshot = {}
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None:
                    snapshot[(ws.title, cell.coordinate)] = cell.value
    return snapshot


def _outside_value_changes(before: dict, after: dict, allowed: set) -> list[dict]:
    return [
        {"sheet": key[0], "cell": key[1], "before": str(before.get(key)),
         "after": str(after.get(key))}
        for key in sorted(set(before) | set(after))
        if key not in allowed and before.get(key) != after.get(key)
    ]


def apply_edits(src_path: Path, edits: list[dict], *, dry_run: bool = False,
                allow_formula: bool = False, allow_merged: bool = False,
                allow_protected: bool = False, allow_hidden: bool = False) -> dict:
    wb = openpyxl.load_workbook(src_path)  # 서식 유지 위해 data_only 미사용
    planned, errors = _preflight_edits(
        wb, edits,
        allow_formula=allow_formula,
        allow_merged=allow_merged,
        allow_protected=allow_protected,
        allow_hidden=allow_hidden,
    )

    if errors:
        # 하나라도 실패하면 파일을 쓰지 않고 중단 (부분 적용 방지)
        return {"status": "aborted", "errors": errors, "applied": []}
    if dry_run:
        return {"status": "preview", "source": str(src_path), "planned": planned}

    source_structure = _structure_snapshot(wb)
    source_values = _value_snapshot(wb)
    change_log = []
    for item in planned:
        ws = wb[item["sheet"]]
        ws[item["cell"]].value = item["new_value"]
        change_log.append({
            **item,
            "old_value": None if item["old_value"] is None else str(item["old_value"]),
            "new_value": str(item["new_value"]),
        })

    # 타임스탬프 복사본 경로 (원본은 그대로 둔다)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = src_path.with_name(f"{src_path.stem}_revised_{ts}{src_path.suffix}")

    # atomic write: .tmp 로 먼저 저장 후 os.replace()
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    wb.save(tmp_path)
    os.replace(tmp_path, out_path)

    # 저장 결과 재검증: 허가된 대상 셀의 값이 실제로 반영됐는지 확인한다.
    # 구조/병합 범위까지 검증해야 하므로 read_only 모드를 쓰지 않는다.
    verify_wb = openpyxl.load_workbook(out_path, read_only=False, data_only=False)
    verification_structure = _structure_snapshot(verify_wb)
    verification_values = _value_snapshot(verify_wb)
    verification_errors = []
    for item in planned:
        actual = verify_wb[item["sheet"]][item["cell"]].value
        if actual != item["new_value"]:
            verification_errors.append({
                "sheet": item["sheet"], "cell": item["cell"],
                "expected": item["new_value"], "actual": actual,
            })
    verify_wb.close()
    allowed = {(item["sheet"], item["cell"]) for item in planned}
    outside_changes = _outside_value_changes(source_values, verification_values, allowed)
    structure_changed = source_structure != verification_structure
    if verification_errors or outside_changes or structure_changed:
        return {
            "status": "verification_failed", "source": str(src_path),
            "revised": str(out_path), "errors": verification_errors,
            "outside_value_changes": outside_changes,
            "structure_changed": structure_changed,
        }

    # 변경 로그도 atomic write 로 함께 기록
    log_path = out_path.with_suffix(".changes.json")
    log_payload = {
        "source": str(src_path),
        "revised": str(out_path),
        "timestamp": ts,
        "changes": change_log,
        "verification": {
            "target_values_verified": True,
            "outside_value_changes": [],
            "structure_changed": False,
            "rich_text_note": "C열 텍스트 편집은 rich text 하이라이트 재생성이 필요할 수 있음",
        },
    }
    log_tmp = log_path.with_suffix(log_path.suffix + ".tmp")
    log_tmp.write_text(
        json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(log_tmp, log_path)

    return {
        "status": "ok",
        "source": str(src_path),
        "revised": str(out_path),
        "change_log": str(log_path),
        "applied": change_log,
    }


def main():
    parser = argparse.ArgumentParser(
        description="승인된 편집을 워크북 복사본에 적용 (원본 불변, atomic write)"
    )
    parser.add_argument("workbook", help="원본 .xlsx 경로 (수정되지 않음)")
    parser.add_argument("edits", help="edits JSON 파일 경로 또는 inline JSON 문자열")
    parser.add_argument("--app-root", help="app repo 경로 명시 (Excel-only 에선 불필요)")
    parser.add_argument("--dry-run", action="store_true", help="변경 preview만 출력하고 파일을 쓰지 않음")
    parser.add_argument("--allow-formula", action="store_true", help="수식 셀 수정 허용 (승인 후에만)")
    parser.add_argument("--allow-merged", action="store_true", help="병합 범위 셀 수정 허용 (승인 후에만)")
    parser.add_argument("--allow-protected", action="store_true", help="보호 시트 수정 허용 (승인 후에만)")
    parser.add_argument("--allow-hidden", action="store_true", help="숨김 시트 수정 허용 (승인 후에만)")
    parser.add_argument("--json", action="store_true", help="JSON 출력")
    args = parser.parse_args()

    src_path = Path(args.workbook).expanduser()
    if not src_path.is_file():
        msg = {"error": f"파일을 찾을 수 없습니다: {src_path}"}
        print(json.dumps(msg, ensure_ascii=False) if args.json else f"❌ {msg['error']}")
        sys.exit(2)

    try:
        edits = _load_edits(args.edits)
    except (json.JSONDecodeError, ValueError) as e:
        msg = {"error": f"edits 파싱 실패: {e}"}
        print(json.dumps(msg, ensure_ascii=False) if args.json else f"❌ {msg['error']}")
        sys.exit(2)

    res = apply_edits(
        src_path, edits,
        dry_run=args.dry_run,
        allow_formula=args.allow_formula,
        allow_merged=args.allow_merged,
        allow_protected=args.allow_protected,
        allow_hidden=args.allow_hidden,
    )

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        if res["status"] == "preview":
            print(f"🔎 변경 preview (원본 불변): {res['source']}")
            for c in res["planned"]:
                print(f"   [{c['sheet']}!{c['cell']}] {c['old_value']!r} → {c['new_value']!r}")
        elif res["status"] == "ok":
            print(f"✅ 수정본 생성: {res['revised']}")
            print(f"   변경 로그  : {res['change_log']}")
            print(f"   원본 (불변): {res['source']}")
            for c in res["applied"]:
                print(f"   [{c['sheet']}!{c['cell']}] "
                      f"{c['old_value']!r} → {c['new_value']!r}")
        elif res["status"] == "aborted":
            print(f"❌ 적용 중단 ({len(res['errors'])}건 오류, 파일 미생성):")
            for e in res["errors"]:
                print(f"   #{e['index']}: {e['error']}")
            sys.exit(1)
        else:
            print(f"❌ 저장 후 검증 실패: {res['revised']}")
            for e in res["errors"]:
                print(f"   [{e['sheet']}!{e['cell']}] 기대값={e['expected']!r}, 실제값={e['actual']!r}")
            if res["outside_value_changes"]:
                print(f"   승인 범위 밖 값 변경: {len(res['outside_value_changes'])}건")
            if res["structure_changed"]:
                print("   워크북 구조 변경 감지")
            sys.exit(1)


if __name__ == "__main__":
    main()
