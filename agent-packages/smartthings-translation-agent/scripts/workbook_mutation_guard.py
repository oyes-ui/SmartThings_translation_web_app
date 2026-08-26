#!/usr/bin/env python3
"""Canonical safety primitives for SmartThings one-off Excel mutation tools.

New workbook writers should import this module instead of reimplementing rich
text cloning, cross-workbook layout copying, physical row deletion, semantic
fingerprints, or verified atomic saves.  Extend this module with tests when a
new mutation needs a capability that is not yet represented here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import openpyxl
from openpyxl.cell.rich_text import CellRichText, TextBlock


def text(value: Any) -> str:
    return "" if value is None else str(value)


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clone_rich_value(value: Any) -> Any:
    """Clone a cell value without flattening rich-text runs."""
    if isinstance(value, CellRichText):
        result = CellRichText()
        for part in value:
            if isinstance(part, TextBlock):
                result.append(TextBlock(text=part.text or "", font=copy.copy(part.font)))
            else:
                result.append(str(part))
        return result
    return copy.copy(value)


def copy_cell_style(source, target) -> None:
    """Copy semantic style components safely across different workbooks."""
    target.font = copy.copy(source.font)
    target.fill = copy.copy(source.fill)
    target.border = copy.copy(source.border)
    target.alignment = copy.copy(source.alignment)
    target.protection = copy.copy(source.protection)
    target.number_format = source.number_format


def copy_row_layout(
    source_sheet,
    target_sheet,
    row: int,
    *,
    columns: Iterable[int] | None = None,
) -> None:
    """Copy row geometry and cell styles without copying cell values."""
    source_dimension = source_sheet.row_dimensions.get(row)
    if source_dimension is None:
        if row in target_sheet.row_dimensions:
            del target_sheet.row_dimensions[row]
    else:
        target_dimension = target_sheet.row_dimensions[row]
        for attribute in (
            "height", "hidden", "outlineLevel", "collapsed", "thickTop", "thickBot",
        ):
            setattr(target_dimension, attribute, copy.copy(getattr(source_dimension, attribute)))
    selected = columns if columns is not None else range(1, source_sheet.max_column + 1)
    for column in selected:
        copy_cell_style(source_sheet.cell(row, column), target_sheet.cell(row, column))


def delete_rows_with_manifest(sheet, start: int, amount: int) -> dict[str, Any]:
    """Physically delete rows and return every removed non-empty cell for audit."""
    if start < 1 or amount < 0:
        raise ValueError("행 삭제 범위가 올바르지 않습니다")
    if amount == 0:
        return {"start": start, "end": start - 1, "rows": [], "values": {}}
    end = start + amount - 1
    values = {
        cell.coordinate: text(cell.value)
        for row in sheet.iter_rows(min_row=start, max_row=end)
        for cell in row
        if cell.value is not None
    }
    styled_cells = {
        cell.coordinate: _hash_json(_style_signature(cell))
        for row in sheet.iter_rows(min_row=start, max_row=end)
        for cell in row
        if cell.has_style
    }
    deleted_dimensions = {
        str(index): _dimension_signature(dimension)
        for index, dimension in sheet.row_dimensions.items()
        if start <= index <= end
    }

    relocated_merges: list[tuple[str, str | None]] = []
    for merged in list(sheet.merged_cells.ranges):
        if merged.max_row < start:
            continue
        if merged.min_row > end:
            old = str(merged)
            new = (
                f"{sheet.cell(merged.min_row - amount, merged.min_col).coordinate}:"
                f"{sheet.cell(merged.max_row - amount, merged.max_col).coordinate}"
            )
            relocated_merges.append((old, new))
            continue
        if merged.min_row >= start and merged.max_row <= end:
            relocated_merges.append((str(merged), None))
            continue
        raise ValueError(f"삭제 행 경계를 가로지르는 병합 셀이 있습니다: {sheet.title}!{merged}")

    sheet.delete_rows(start, amount)

    # openpyxl moves cells but not RowDimension objects or merged ranges.
    dimensions = [(index, copy.copy(value)) for index, value in sheet.row_dimensions.items()]
    for index in list(sheet.row_dimensions):
        del sheet.row_dimensions[index]
    for index, dimension in dimensions:
        if start <= index <= end:
            continue
        new_index = index - amount if index > end else index
        dimension.index = new_index
        sheet.row_dimensions[new_index] = dimension

    for old, new in relocated_merges:
        sheet.unmerge_cells(old)
        if new is not None:
            sheet.merge_cells(new)

    return {
        "start": start,
        "end": end,
        "rows": list(range(start, end + 1)),
        "values": values,
        "styled_cells": styled_cells,
        "row_dimensions": deleted_dimensions,
        "merged_ranges": [{"before": old, "after": new} for old, new in relocated_merges],
    }


def _style_signature(cell) -> dict[str, Any]:
    return {
        "font": str(cell.font),
        "fill": str(cell.fill),
        "border": str(cell.border),
        "alignment": str(cell.alignment),
        "protection": str(cell.protection),
        "number_format": cell.number_format,
    }


def _rich_signature(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, CellRichText):
        return None
    result: list[dict[str, str]] = []
    for part in value:
        if isinstance(part, TextBlock):
            result.append({
                "text_sha256": hashlib.sha256((part.text or "").encode("utf-8")).hexdigest(),
                "font": str(part.font),
            })
        else:
            result.append({
                "text_sha256": hashlib.sha256(str(part).encode("utf-8")).hexdigest(),
                "font": "",
            })
    return result


def _dimension_signature(dimension) -> dict[str, Any]:
    return {
        "height": getattr(dimension, "height", None),
        "width": getattr(dimension, "width", None),
        "hidden": getattr(dimension, "hidden", False),
        "outlineLevel": getattr(dimension, "outlineLevel", 0),
        "collapsed": getattr(dimension, "collapsed", False),
        "thickTop": getattr(dimension, "thickTop", False),
        "thickBot": getattr(dimension, "thickBot", False),
    }


def semantic_workbook_snapshot(workbook) -> dict[str, Any]:
    """Return independent hashes for values, layout, styles, and rich text."""
    values: list[dict[str, str]] = []
    styles: list[dict[str, Any]] = []
    rich_text: list[dict[str, Any]] = []
    annotations: list[dict[str, str]] = []
    sheets: dict[str, Any] = {}

    for sheet in workbook.worksheets:
        sheets[sheet.title] = {
            "state": sheet.sheet_state,
            "max_row": sheet.max_row,
            "max_column": sheet.max_column,
            "merged_ranges": sorted(str(value) for value in sheet.merged_cells.ranges),
            "freeze_panes": str(sheet.freeze_panes) if sheet.freeze_panes else None,
            "auto_filter": sheet.auto_filter.ref,
            "tab_color": str(sheet.sheet_properties.tabColor) if sheet.sheet_properties.tabColor else None,
            "row_dimensions": {
                str(index): _dimension_signature(dimension)
                for index, dimension in sorted(sheet.row_dimensions.items())
            },
            "column_dimensions": {
                str(index): _dimension_signature(dimension)
                for index, dimension in sorted(sheet.column_dimensions.items())
            },
            "data_validations": sorted(str(item.sqref) for item in sheet.data_validations.dataValidation),
            "image_count": len(getattr(sheet, "_images", [])),
            "chart_count": len(getattr(sheet, "_charts", [])),
        }
        for (_row, _column), cell in sorted(sheet._cells.items()):
            coordinate = cell.coordinate
            if cell.value is not None:
                values.append({
                    "sheet": sheet.title,
                    "cell": coordinate,
                    "value_sha256": hashlib.sha256(text(cell.value).encode("utf-8")).hexdigest(),
                    "data_type": cell.data_type,
                })
            if cell.has_style:
                styles.append({"sheet": sheet.title, "cell": coordinate, **_style_signature(cell)})
            rich = _rich_signature(cell.value)
            if rich is not None:
                rich_text.append({"sheet": sheet.title, "cell": coordinate, "runs": rich})
            if cell.comment is not None:
                annotations.append({
                    "sheet": sheet.title,
                    "cell": coordinate,
                    "comment_sha256": hashlib.sha256(cell.comment.text.encode("utf-8")).hexdigest(),
                    "author_sha256": hashlib.sha256((cell.comment.author or "").encode("utf-8")).hexdigest(),
                })

    return {
        "sheet_count": len(workbook.sheetnames),
        "values_sha256": _hash_json(values),
        "layout_sha256": _hash_json({"sheetnames": list(workbook.sheetnames), "sheets": sheets}),
        "styles_sha256": _hash_json(styles),
        "rich_text_sha256": _hash_json(rich_text),
        "annotations_sha256": _hash_json(annotations),
    }


def snapshot_path(path: str | Path) -> dict[str, Any]:
    workbook = openpyxl.load_workbook(path, data_only=False, rich_text=True)
    try:
        return semantic_workbook_snapshot(workbook)
    finally:
        workbook.close()


def snapshot_axis_diff(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    axes = ("values_sha256", "layout_sha256", "styles_sha256", "rich_text_sha256", "annotations_sha256")
    return [axis.removesuffix("_sha256") for axis in axes if before.get(axis) != after.get(axis)]


def save_verified_atomic(
    workbook,
    output: str | Path,
    verify_path: Callable[[Path], Any],
    *,
    overwrite: bool = False,
) -> Any:
    """Save to a temp file, verify the reopened temp, then atomically promote."""
    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.verify-",
        suffix=destination.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        workbook.save(temporary)
        result = verify_path(temporary)
        os.replace(temporary, destination)
        return result
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
