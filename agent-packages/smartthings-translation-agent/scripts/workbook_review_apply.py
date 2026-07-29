#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply an approved native-review decision manifest to a workbook copy.

This is deliberately an executor, not a reviewer.  It reads the current text
from C, reviewer proposals from F and comments from H, but changes C only for
``accept`` / ``partial`` decisions already recorded in a manifest.  The source
workbook is never modified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openpyxl
from openpyxl.utils.cell import column_index_from_string, coordinate_from_string

import _app_pipeline as ap
from workbook_highlight_glossary import _report_path_for, run_highlight


DECISIONS = {"accept", "partial", "hold"}
DEFAULT_PROTECTED = ["KR(한국)", "US(미국)", "CN(중국)", "BR(브라질)", "RU(러시아)"]


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("decisions"), list):
        raise ValueError("판정 manifest는 decisions 배열을 가진 JSON 객체여야 합니다.")
    return data


def _parse_c_range(raw: str) -> tuple[int, int]:
    match = re.fullmatch(r"C(\d+):C(\d+)", raw.strip().upper())
    if not match:
        raise ValueError("--cell-range는 C7:C28처럼 C열의 연속 범위여야 합니다.")
    start, end = map(int, match.groups())
    if start > end:
        raise ValueError("--cell-range의 시작 행이 끝 행보다 큽니다.")
    return start, end


def _parse_columns(raw: str) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z]+):([A-Z]+)", raw.strip().upper())
    if not match:
        raise ValueError("--drop-review-columns는 E:H처럼 열 범위여야 합니다.")
    start, end = (column_index_from_string(item) for item in match.groups())
    if start > end:
        raise ValueError("삭제할 열 범위가 올바르지 않습니다.")
    return start, end


def _source_for_sheet(sheet: str) -> str:
    return "KR(한국)" if sheet in ap.GROUP_A_TARGETS else "US(미국)"


def _validate_decisions(
    wb: openpyxl.Workbook,
    decisions: list[dict],
    protected: set[str],
    cell_range: str,
) -> list[dict]:
    start, end = _parse_c_range(cell_range)
    seen: set[tuple[str, str]] = set()
    normalized: list[dict] = []

    for index, raw in enumerate(decisions):
        if not isinstance(raw, dict):
            raise ValueError(f"decisions[{index}]는 객체여야 합니다.")
        sheet = str(raw.get("sheet", "")).strip()
        cell = str(raw.get("cell", "")).strip().upper()
        decision = str(raw.get("decision", "")).strip().lower()
        if sheet not in wb.sheetnames:
            raise ValueError(f"decisions[{index}]: 없는 시트 {sheet!r}")
        if decision not in DECISIONS:
            raise ValueError(f"decisions[{index}]: decision은 {sorted(DECISIONS)} 중 하나여야 합니다.")
        try:
            col, row = coordinate_from_string(cell)
        except ValueError as exc:
            raise ValueError(f"decisions[{index}]: 잘못된 셀 좌표 {cell!r}") from exc
        if col != "C" or not start <= row <= end:
            raise ValueError(f"decisions[{index}]: {cell}은 허용 범위 {cell_range} 밖입니다.")
        key = (sheet, cell)
        if key in seen:
            raise ValueError(f"같은 셀의 판정이 중복되었습니다: {sheet}!{cell}")
        seen.add(key)
        if decision in {"accept", "partial"} and sheet in protected:
            raise ValueError(f"보호 언어는 수용/부분 수용할 수 없습니다: {sheet}!{cell}")
        declared_source = raw.get("source_sheet")
        if declared_source and declared_source != _source_for_sheet(sheet):
            raise ValueError(
                f"{sheet}!{cell}: source_sheet={declared_source!r}가 표준 source group "
                f"{_source_for_sheet(sheet)!r}과 다릅니다."
            )
        normalized.append({**raw, "sheet": sheet, "cell": cell, "decision": decision})
    return normalized


def _logical_cells(ws, deleted: tuple[int, int] | None) -> dict[tuple[int, int], tuple[str, Any]]:
    """Snapshot values/formulas after logically removing a column interval."""
    values: dict[tuple[int, int], tuple[str, Any]] = {}
    start, end = deleted or (0, -1)
    width = end - start + 1
    for row in ws.iter_rows():
        for cell in row:
            if start <= cell.column <= end:
                continue
            logical_col = cell.column - width if cell.column > end else cell.column
            if cell.value is not None:
                values[(cell.row, logical_col)] = (cell.data_type, cell.value)
    return values


def _logical_merges(ws, deleted: tuple[int, int] | None) -> set[str]:
    start, end = deleted or (0, -1)
    width = end - start + 1
    result: set[str] = set()
    for merged in ws.merged_cells.ranges:
        min_col, min_row, max_col, max_row = merged.bounds
        if not (max_col < start or min_col > end):
            continue
        if min_col > end:
            min_col -= width
            max_col -= width
        result.add(f"{min_col}:{min_row}:{max_col}:{max_row}")
    return result


def _snapshot_protected(wb, protected: set[str], deleted: tuple[int, int] | None = None) -> dict:
    return {
        name: {
            "cells": _logical_cells(wb[name], deleted),
            "merges": _logical_merges(wb[name], deleted),
            "c_values": [_text(wb[name][f"C{row}"].value) for row in range(1, wb[name].max_row + 1)],
        }
        for name in protected
    }


def _verify_protected(before: dict, after: dict) -> dict:
    failed: list[str] = []
    for sheet, snapshot in before.items():
        if snapshot["c_values"] != after[sheet]["c_values"]:
            failed.append(f"{sheet}: C열 값")
        if snapshot["cells"] != after[sheet]["cells"]:
            failed.append(f"{sheet}: 값/수식")
        if snapshot["merges"] != after[sheet]["merges"]:
            failed.append(f"{sheet}: 병합")
    if failed:
        raise RuntimeError("보호 언어 무결성 검증 실패: " + ", ".join(failed))
    return {"passed": True, "sheets": sorted(before)}


def _apply_decisions(wb, decisions: list[dict]) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    changes: list[dict] = []
    for item in decisions:
        ws = wb[item["sheet"]]
        current = _text(ws[item["cell"]].value)
        reviewer_cell = str(item.get("reviewer_cell") or f"F{item['cell'][1:]}").upper()
        reviewer = _text(ws[reviewer_cell].value)
        comment_cell = str(item.get("comment_cell") or f"H{item['cell'][1:]}").upper()
        comment = _text(ws[comment_cell].value)
        decision = item["decision"]
        if decision == "accept":
            if not reviewer or reviewer == current:
                raise ValueError(f"{item['sheet']}!{item['cell']}: 수용할 F열 수정안이 없거나 현재값과 같습니다.")
            final = reviewer
        elif decision == "partial":
            if "final_value" not in item or not _text(item["final_value"]):
                raise ValueError(f"{item['sheet']}!{item['cell']}: 부분 수용에는 final_value가 필요합니다.")
            final = _text(item["final_value"])
            if final == current:
                raise ValueError(f"{item['sheet']}!{item['cell']}: 부분 수용 final_value가 현재값과 같습니다.")
        else:
            final = current
        entry = {
            "sheet": item["sheet"], "cell": item["cell"], "decision": decision,
            "current": current, "reviewer_proposal": reviewer, "reviewer_comment": comment,
            "final": final, "reason": item.get("reason", ""), "basis": item.get("basis", ""),
            "rag_basis": item.get("rag_basis", ""), "source_sheet": item.get("source_sheet", _source_for_sheet(item["sheet"])),
        }
        if decision != "hold":
            ws[item["cell"]].value = final
            changes.append(entry)
        records.append(entry)
    return records, changes


def _verify_expected_c_diff(source: Path, acceptance: Path, expected: set[tuple[str, str]]) -> dict:
    before = openpyxl.load_workbook(source, data_only=False)
    after = openpyxl.load_workbook(acceptance, data_only=False)
    actual: set[tuple[str, str]] = set()
    for name in before.sheetnames:
        for row in range(1, max(before[name].max_row, after[name].max_row) + 1):
            if _text(before[name][f"C{row}"].value) != _text(after[name][f"C{row}"].value):
                actual.add((name, f"C{row}"))
    if actual != expected:
        raise RuntimeError(f"C열 값 diff 불일치: unexpected={sorted(actual - expected)}, missing={sorted(expected - actual)}")
    return {"expected": len(expected), "actual": len(actual), "passed": True}


async def run_apply(args) -> dict:
    source = Path(args.workbook).expanduser()
    approval_path = Path(args.approval_manifest).expanduser()
    output = Path(args.output).expanduser()
    glossary = Path(args.glossary).expanduser()
    if not source.is_file() or not approval_path.is_file() or not glossary.is_file():
        raise FileNotFoundError("workbook, approval manifest, glossary 경로를 모두 확인하세요.")
    # Validate/re-exec into the app runtime before creating any acceptance copy.
    app_root = ap.bootstrap_project(args.app_root)
    ap.maybe_reexec_with_app_venv(app_root)
    manifest = _load_manifest(approval_path)
    protected = set(manifest.get("protected_sheets", args.protected_sheets.split(",")))
    wb = openpyxl.load_workbook(source, data_only=False)
    missing_protected = sorted(protected - set(wb.sheetnames))
    if missing_protected:
        raise ValueError("워크북에 없는 보호 시트: " + ", ".join(missing_protected))
    decisions = _validate_decisions(wb, manifest["decisions"], protected, args.cell_range)
    deleted = _parse_columns(args.drop_review_columns) if args.drop_review_columns else None
    protected_before = _snapshot_protected(wb, protected, deleted)
    decision_records, changes = _apply_decisions(wb, decisions)

    if deleted:
        start, end = deleted
        for ws in wb.worksheets:
            ws.delete_cols(start, end - start + 1)

    protected_after = _snapshot_protected(wb, protected)
    protected_validation = _verify_protected(protected_before, protected_after)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    wb.save(temp)
    os.replace(temp, output)
    expected = {(change["sheet"], change["cell"]) for change in changes}
    diff_validation = _verify_expected_c_diff(source, output, expected)

    highlight_args = SimpleNamespace(
        app_root=str(app_root), workbook=str(output), glossary=str(glossary), sheets=None,
        cell_range=args.cell_range, sheet_langs=None, single_source=False, source_sheet="US(미국)",
        include_source_sheets=True, max_concurrency=args.max_concurrency, json=True, verbose=False,
    )
    highlighted = await run_highlight(highlight_args)
    if highlighted.get("status") != "ok" or not highlighted.get("excel_path"):
        raise RuntimeError(f"전체 glossary 하이라이트 실패: {highlighted.get('errors') or highlighted.get('error')}")
    final = Path(highlighted["excel_path"])
    if _verify_expected_c_diff(output, final, set())["actual"] != 0:
        raise RuntimeError("하이라이트가 C열 문안을 변경했습니다.")
    report_path = _report_path_for(highlighted)
    result = {
        "status": "ok", "source": str(source), "approval_manifest": str(approval_path),
        "acceptance_copy": str(output), "final": str(final), "glossary": str(glossary),
        "cell_range": args.cell_range, "review_columns_removed": args.drop_review_columns or None,
        "decisions": decision_records, "changes": changes, "decision_counts": {
            "accept": sum(d["decision"] == "accept" for d in decisions),
            "partial": sum(d["decision"] == "partial" for d in decisions),
            "hold": sum(d["decision"] == "hold" for d in decisions),
        },
        "value_diff_validation": diff_validation, "protected_validation": protected_validation,
        "highlight_validation": highlighted.get("text_validation"), "highlight_report": report_path,
    }
    result_path = Path(args.result_manifest).expanduser() if args.result_manifest else final.with_suffix(".review_apply.json")
    result["result_manifest"] = str(result_path)
    _atomic_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="승인된 감수 판정만 반영해 최종 하이라이트 납품본 생성")
    parser.add_argument("workbook", help="감수본 .xlsx (원본 불변)")
    parser.add_argument("approval_manifest", help="수용/부분 수용/유지 판정 JSON")
    parser.add_argument("--output", required=True, help="하이라이트 전 1차 수용본 .xlsx")
    parser.add_argument("--glossary", required=True, help="기준 glossary CSV")
    parser.add_argument("--cell-range", default="C7:C28")
    parser.add_argument("--protected-sheets", default=",".join(DEFAULT_PROTECTED))
    parser.add_argument("--drop-review-columns", help="명시 시 감수 열 삭제. 예: E:H")
    parser.add_argument("--result-manifest", help="생성 결과 manifest 경로")
    parser.add_argument("--app-root", help="SmartThings app repo")
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = asyncio.run(run_apply(args))
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
