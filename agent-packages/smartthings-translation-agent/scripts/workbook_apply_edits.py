#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workbook_apply_edits.py — 승인된 편집을 워크북에 적용 (원본 불변)

⚠ 이 스크립트는 사용자가 명시적으로 승인한 뒤에만 실행해야 한다.

안전 정책:
  - 원본 파일은 절대 수정하지 않는다. 항상 검증된 버전 폴더의 '복사본'에 기록한다.
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
from openpyxl.cell.rich_text import CellRichText  # noqa: E402
from openpyxl.utils import get_column_letter, column_index_from_string  # noqa: E402
from openpyxl.utils.cell import coordinate_from_string  # noqa: E402
from workbook_manifest import create_edit_revision, resolve_ledger  # noqa: E402


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
        # report_format_spec manifest는 workbook_review_apply.py가 직접 변환한다.
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
            comparable = str(old_value) if isinstance(old_value, CellRichText) else old_value
            if "before" in edit and comparable != edit["before"]:
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
                "reason": edit.get("reason"),
                "rule_ids": edit.get("rule_ids", []),
            })
        except Exception as e:
            errors.append({"index": i, "edit": edit, "error": str(e)})
    return planned, errors


def apply_edits(src_path: Path, edits: list[dict], *, dry_run: bool = False,
                allow_formula: bool = False, allow_merged: bool = False,
                allow_protected: bool = False, allow_hidden: bool = False) -> dict:
    """Existing authorized edit entry point; preview never grants approval.

    The caller must already have the user's approval to invoke without dry_run.
    Persist the plan on preview so changed inputs cannot inherit that approval.
    """
    from workbook_contract import atomic_json, digest, check_inputs, file_sha256, load_contract
    from workbook_run import exclusive_lock
    source = src_path.resolve()
    options = dict(allow_formula=allow_formula, allow_merged=allow_merged,
                   allow_protected=allow_protected, allow_hidden=allow_hidden)
    request = {"source": str(source), "edits": edits, "options": options}
    work_id = "edit-" + digest(request)[:20]
    root = source.parent / ".st-runs" / work_id
    contract_path = root / "contract.json"
    try:
        with exclusive_lock(root / ".prepare.lock"):
            if not contract_path.exists():
                prepare_edit_contract(source, edits, work_root=root,
                    delivery_root=source.parent / "verified", work_id=work_id, **options)
            contract = load_contract(contract_path)
            if contract.get("request_edits") != edits or contract.get("options") != options:
                raise ValueError("Request differs from the saved plan")
            from workbook_run import _published
            if _published(contract_path, contract) is None:
                check_inputs(contract)
            if dry_run:
                return {"status": "preview", "source": str(source), "contract": str(contract_path),
                        "planned": [{"sheet": x["sheet"], "cell": x["cell"],
                                     "old_value": x["before"], "new_value": x["after"]}
                                    for x in contract["artifacts"][0]["edits"]]}
            approval = root / "approval.json"
            atomic_json(approval, {"approved": True, "approved_by": "authorized_edit_call",
                                  "contract_sha256": file_sha256(contract_path)})
        return apply_edits_contract(source, edits, contract_path, approval)
    except (ValueError, OSError) as error:
        return {"status": "aborted", "errors": [{"index": 0, "error": str(error)}],
                "applied": [], "contract": str(contract_path)}


def contract_writer_identity() -> dict:
    from workbook_contract import file_sha256
    return {"id": "workbook_apply_edits", "version": file_sha256(Path(__file__)), "cost": "local"}


def prepare_edit_contract(src_path: Path, edits: list[dict], *, work_root: Path,
                          delivery_root: Path, work_id: str, persist: bool = True, **flags) -> dict:
    """Read-only workbook plan. Writes only the reviewable contract, never approves it."""
    from openpyxl.cell.cell import Cell
    from openpyxl.compat import safe_string
    from workbook_contract import (ContractError, capture_authority, canonical,
                                    save_contract)
    from workbook_verifier import facts, slot
    source = src_path.resolve()
    authority = capture_authority(source)
    baseline = facts(source)
    book = openpyxl.load_workbook(source, rich_text=True, data_only=False)
    try:
        options = {name: bool(flags.get(name, False)) for name in
                   ("allow_formula", "allow_merged", "allow_protected", "allow_hidden")}
        planned, errors = _preflight_edits(book, edits, **options)
        if errors:
            raise ContractError(str(errors))
        seen, allowed, normalized = set(), [], []
        for item in planned:
            key = canonical(["sheets", item["sheet"], "cells", item["cell"]])
            if key in seen:
                raise ContractError("Duplicate cell edits are not supported")
            seen.add(key)
            current = book[item["sheet"]][item["cell"]]
            if current.row > baseline["layout"][canonical(["sheets", item["sheet"], "max_row"])] or current.column > baseline["layout"][canonical(["sheets", item["sheet"], "max_column"])]:
                raise ContractError("New rows/columns require a structural contract")
            value = item["new_value"]
            if value is not None and type(value) not in (str, bool, int, float):
                raise ContractError("Pilot edits accept JSON scalar cell values only")
            if value == "":
                raise ContractError("Use null for an empty cell (Excel empty-string normalization)")
            cell = Cell(book[item["sheet"]], value=value)
            stored = value
            if type(value) in (int, float):
                encoded = safe_string(value)
                if not encoded:
                    raise ContractError("Excel cannot store non-finite numbers")
                stored = float(encoded) if any(x in encoded for x in (".", "e", "E")) else int(encoded)
                if stored != value:
                    raise ContractError("Numeric value would lose precision in Excel")
            if cell.value != value:
                raise ContractError("Excel would truncate or normalize this cell value")
            after = ({"present": False} if value is None else
                     {"present": True, "value": {"text": str(stored), "data_type": cell.data_type}})
            before = slot(baseline["values"], key)
            if before != after:
                allowed.append({"axis": "values", "path": json.loads(key), "before": before, "after": after})
            if key in baseline["rich_text"]:
                allowed.append({"axis": "rich_text", "path": json.loads(key),
                                "before": slot(baseline["rich_text"], key), "after": {"present": False}})
            normalized.append({"sheet": item["sheet"], "cell": item["cell"],
                               "after": value, "before": None if item["old_value"] is None else str(item["old_value"]),
                               "reason": item.get("reason"), "rule_ids": item.get("rule_ids", [])})
    finally:
        book.close()
    reference = {"authority": "source", "file": "."}
    contract = {
        "schema_version": 1, "work_id": work_id,
        "work_root": str(work_root.resolve()), "staging_root": str((work_root / "staging").resolve()),
        "delivery_root": str(delivery_root.resolve()),
        "authorities": {"source": authority}, "writer": contract_writer_identity(),
        "request_edits": edits, "options": options,
        "artifacts": [{"id": "edited", "output": source.stem + "_revised.xlsx",
                       "authorities": {role: dict(reference) for role in ("values", "structure", "formatting")},
                       "allowed_diffs": allowed, "edits": normalized}],
        "no_op_files": ["edited"] if not allowed else [],
        "recovery": {"max_attempts": 3, "same_failure_limit": 2},
    }
    if not persist:
        return contract
    path = work_root / "contract.json"
    sha = save_contract(path, contract)
    return {"status": "preview", "contract": str(path.resolve()), "contract_sha256": sha,
            "planned": normalized, "allowed_diffs": allowed,
            "approval_required": {"approved": True, "approved_by": "<user>", "contract_sha256": sha}}


def _build_contract_edits(contract: dict, artifact: dict, last_failure: dict | None):
    from workbook_contract import authority_file
    book = openpyxl.load_workbook(authority_file(contract, artifact["authorities"]["values"]),
                                  rich_text=True, data_only=False)
    try:
        # Preflight against original request preserves hidden/protected/formula gates.
        planned, errors = _preflight_edits(book, contract["request_edits"], **contract["options"])
        if errors:
            raise ValueError(str(errors))
        for item in planned:
            book[item["sheet"]][item["cell"]].value = item["new_value"]
        return book
    except BaseException:
        book.close()
        raise


def apply_edits_contract(src_path: Path, edits: list[dict], contract_path: Path,
                         approval_path: Path | None = None) -> dict:
    from workbook_contract import ContractError, atomic_json, authority_file, load_contract
    from workbook_run import execute_run, exclusive_lock
    contract = load_contract(contract_path)
    source = authority_file(contract, contract["artifacts"][0]["authorities"]["values"])
    if source.resolve() != src_path.resolve() or edits != contract.get("request_edits"):
        raise ContractError("Request does not match the frozen contract")
    state = execute_run(contract_path, _build_contract_edits, writer=contract_writer_identity(),
                        approval_path=approval_path)
    if state["status"] != "completed":
        return state
    root = Path(contract["work_root"])
    # Existing revision/change-log formats stay intact. A completed run reuses them.
    with exclusive_lock(root / ".run.lock"):
        cached = root / "edit_result.json"
        revised = Path(state["result"]["outputs"]["edited"])
        if cached.exists():
            result = json.loads(cached.read_text())
            if result.get("status") != "ok" or result.get("revised") != str(revised) or result.get("source") != str(source):
                raise ContractError("Cached edit result does not match the verified publication")
            if not Path(result["revision_manifest"]).is_file() or not Path(result["change_log"]).is_file():
                raise ContractError("Edit history sidecars are missing")
            return result
        changes = [{"sheet": x["sheet"], "cell": x["cell"], "old_value": x["before"],
                    "new_value": x["after"], "reason": x["reason"], "rule_ids": x["rule_ids"]}
                   for x in contract["artifacts"][0]["edits"]]
        revision, revision_path = create_edit_revision(source, revised, changes)
        log_path = revised.with_suffix(".changes.json")
        atomic_json(log_path, {"source": str(source), "revised": str(revised), "changes": changes,
                              "revision_id": revision["revision_id"], "revision_manifest": str(revision_path),
                              "verification": state["result"]["records"]["edited"]})
        result = {"status": "ok", "artifact_status": "draft", "source": str(source), "revised": str(revised),
                  "change_log": str(log_path), "revision_manifest": str(revision_path),
                  "revision_id": revision["revision_id"], "applied": changes,
                  "run_state": str(root / "state.json"), "batch_manifest": str(revised.parent / "batch.complete.json")}
        atomic_json(cached, result)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="승인된 편집을 워크북 복사본에 적용 (원본 불변, atomic write)"
    )
    parser.add_argument("workbook", help="원본 .xlsx 경로 (수정되지 않음)")
    parser.add_argument("edits", help="edits JSON 파일 경로 또는 inline JSON 문자열")
    parser.add_argument("--app-root", help="app repo 경로 명시 (Excel-only 에선 불필요)")
    parser.add_argument("--dry-run", action="store_true", help="계약 preview만 저장; Excel은 쓰지 않음")
    parser.add_argument("--allow-formula", action="store_true", help="수식 셀 수정 허용 (승인 후에만)")
    parser.add_argument("--allow-merged", action="store_true", help="병합 범위 셀 수정 허용 (승인 후에만)")
    parser.add_argument("--allow-protected", action="store_true", help="보호 시트 수정 허용 (승인 후에만)")
    parser.add_argument("--allow-hidden", action="store_true", help="숨김 시트 수정 허용 (승인 후에만)")
    parser.add_argument("--prepare-run", type=Path, help="계약 preview를 저장할 새 작업 폴더")
    parser.add_argument("--delivery-root", type=Path, help="계약 경로의 검증된 배치 출력 root")
    parser.add_argument("--work-id", help="새 계약의 고유 작업 ID")
    parser.add_argument("--run-contract", type=Path, help="저장된 계약 실행/재개")
    parser.add_argument("--approval", type=Path, help="계약 SHA-256에 결합된 사용자 승인 JSON")
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

    if args.prepare_run or args.run_contract:
        if args.prepare_run and args.run_contract:
            parser.error("prepare-run and run-contract are mutually exclusive")
        try:
            if args.prepare_run:
                if not args.delivery_root or not args.work_id:
                    parser.error("prepare-run requires delivery-root and work-id")
                res = prepare_edit_contract(src_path, edits, work_root=args.prepare_run,
                    delivery_root=args.delivery_root, work_id=args.work_id,
                    allow_formula=args.allow_formula, allow_merged=args.allow_merged,
                    allow_protected=args.allow_protected, allow_hidden=args.allow_hidden)
            elif args.dry_run:
                from workbook_run import read_state
                from workbook_contract import load_contract
                res = {"status": "preview", "state": read_state(load_contract(args.run_contract)["work_root"])}
            else:
                res = apply_edits_contract(src_path, edits, args.run_contract, args.approval)
        except Exception as error:
            res = {"status": "error", "error": str(error)}
        print(json.dumps(res, ensure_ascii=False, indent=2))
        sys.exit(0 if res["status"] in {"ok", "preview", "awaiting_approval"} else 1)

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
            print(json.dumps(res, ensure_ascii=False, indent=2))
    if res["status"] not in {"ok", "preview"}:
        sys.exit(1)


if __name__ == "__main__":
    main()
