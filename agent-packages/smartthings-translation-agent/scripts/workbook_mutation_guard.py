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
import re
import tempfile
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from xml.etree import ElementTree as ET
from zipfile import ZipFile
from pathlib import Path
from typing import Any, Callable, Iterable

import openpyxl
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont

_external_saves = ContextVar("workbook_guard_external_saves", default=False)
_save_patch_lock = threading.RLock()
_save_patch_users = 0
_original_workbook_save = None


@contextmanager
def guard_external_workbook_saves():
    """Route app serializers through this guard while a package adapter runs.

    The hook is context-local (also inherited by asyncio tasks/to_thread).
    Other threads/tasks keep their original save behaviour. Nested/concurrent
    adapters share one hook and the original method is restored on exit.
    """
    global _save_patch_users, _original_workbook_save
    with _save_patch_lock:
        if _save_patch_users == 0:
            _original_workbook_save = openpyxl.Workbook.save
            openpyxl.Workbook.save = _guard_external_save
        _save_patch_users += 1
    token = _external_saves.set(True)
    try:
        yield
    finally:
        _external_saves.reset(token)
        with _save_patch_lock:
            _save_patch_users -= 1
            if _save_patch_users == 0:
                openpyxl.Workbook.save = _original_workbook_save


def _guard_external_save(workbook, output):
    if _external_saves.get():
        return save_workbook_atomic(workbook, output, overwrite=True)
    return _original_workbook_save(workbook, output)


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


def slice_rich_value(value: Any, start: int = 0, end: int | None = None) -> Any:
    """Slice a cell value by displayed-character offsets without flattening runs."""
    displayed = text(value)
    limit = len(displayed)
    normalized_start, normalized_end, _step = slice(start, end).indices(limit)
    if normalized_start >= normalized_end:
        return ""
    if not isinstance(value, CellRichText):
        return displayed[normalized_start:normalized_end]

    result = CellRichText()
    cursor = 0
    for part in value:
        segment = part.text or "" if isinstance(part, TextBlock) else str(part)
        segment_end = cursor + len(segment)
        overlap_start = max(normalized_start, cursor)
        overlap_end = min(normalized_end, segment_end)
        if overlap_start < overlap_end:
            fragment = segment[overlap_start - cursor:overlap_end - cursor]
            if isinstance(part, TextBlock):
                result.append(TextBlock(text=fragment, font=copy.copy(part.font)))
            else:
                result.append(fragment)
        cursor = segment_end
    return result


def replace_span_with_red(
    value: Any, start: int, end: int, replacement: str, base_font: Any,
    *, red_color: str = "FFFF0000",
) -> CellRichText | str:
    """Replace one displayed span while preserving untouched rich-text runs."""
    from workbook_incremental_highlight import _font

    original = text(value)
    if not (0 <= start <= end <= len(original)):
        raise ValueError("Replacement span outside displayed text")
    if not isinstance(replacement, str):
        raise TypeError("replacement must be a string")
    result = CellRichText()

    def append_fragment(fragment: Any) -> None:
        if isinstance(fragment, CellRichText):
            for part in fragment:
                result.append(part)
        elif fragment:
            result.append(str(fragment))

    append_fragment(slice_rich_value(value, 0, start))
    if replacement:
        old_slice = slice_rich_value(value, start, min(start + 1, len(original)))
        if isinstance(old_slice, CellRichText) and old_slice and isinstance(old_slice[0], TextBlock):
            font = copy.copy(old_slice[0].font)
        else:
            font = _font(base_font, None)
        font.color = red_color
        result.append(TextBlock(text=replacement, font=font))
    append_fragment(slice_rich_value(value, end, None))
    expected = original[:start] + replacement + original[end:]
    if str(result) != expected:
        raise ValueError("Rich-text replacement changed unrelated text")
    return result if result else ""


def replace_characters_with_red(
    value: Any, replacements: dict[int, str], base_font: Any,
    *, red_color: str = "FFFF0000",
) -> CellRichText:
    """Replace individual characters and mark only those characters red.

    Unchanged rich-text fragments keep their original runs and inline fonts.
    Character offsets refer to the displayed string, including line breaks.
    """
    from workbook_incremental_highlight import _font

    original = text(value)
    positions = sorted(replacements)
    if any(position < 0 or position >= len(original) for position in positions):
        raise ValueError("Replacement position outside displayed text")
    if any(len(replacements[position]) != 1 for position in positions):
        raise ValueError("Each replacement must be one character")

    output = CellRichText()

    def append_fragment(fragment: Any) -> None:
        if isinstance(fragment, CellRichText):
            for part in fragment:
                output.append(part)
        elif fragment:
            output.append(str(fragment))

    cursor = 0
    for position in positions:
        append_fragment(slice_rich_value(value, cursor, position))
        old_character = slice_rich_value(value, position, position + 1)
        if isinstance(old_character, CellRichText) and isinstance(old_character[0], TextBlock):
            font = copy.copy(old_character[0].font)
        else:
            font = _font(base_font, None)
        font.color = red_color
        output.append(TextBlock(text=replacements[position], font=font))
        cursor = position + 1
    append_fragment(slice_rich_value(value, cursor, None))
    expected = "".join(replacements.get(index, character) for index, character in enumerate(original))
    if str(output) != expected:
        raise ValueError("Rich-text replacement changed unrelated text")
    return output


def insert_text_with_red(
    value: Any, insertions: dict[int, str], base_font: Any,
    *, red_color: str = "FFFF0000",
) -> CellRichText:
    """Insert red text at displayed-character offsets while keeping existing runs."""
    from workbook_incremental_highlight import _font

    original = text(value)
    positions = sorted(insertions)
    if any(position < 0 or position > len(original) for position in positions):
        raise ValueError("Insertion position outside displayed text")
    if any(not isinstance(insertions[position], str) or not insertions[position]
           for position in positions):
        raise ValueError("Each insertion must be nonempty text")

    output = CellRichText()

    def append_fragment(fragment: Any) -> None:
        if isinstance(fragment, CellRichText):
            for part in fragment:
                output.append(part)
        elif fragment:
            output.append(str(fragment))

    cursor = 0
    for position in positions:
        append_fragment(slice_rich_value(value, cursor, position))
        font = _font(base_font, None)
        font.color = red_color
        output.append(TextBlock(text=insertions[position], font=font))
        cursor = position
    append_fragment(slice_rich_value(value, cursor, None))
    expected = original
    for position in reversed(positions):
        expected = expected[:position] + insertions[position] + expected[position:]
    if str(output) != expected:
        raise ValueError("Rich-text insertion changed unrelated text")
    return output


def rewrite_rich_with_blue_spans(value: Any, new_text: str, base_font: Any,
                                 blue_spans: Iterable[tuple[int, int]],
                                 *, blue_color: str = "0432FF",
                                 reset_existing_blue: bool = True) -> CellRichText | str:
    """Keep matched characters' run fonts while replacing text and canonical blue spans.

    This is for review workbooks whose inline fonts contain more than color.
    Existing glossary blue outside the new spans is reset to the cell base color
    by default. Set reset_existing_blue=False for a scoped highlight pass that
    preserves unrelated blue runs. Other run attributes and non-glossary colors
    survive matching characters.
    """
    from difflib import SequenceMatcher
    from workbook_incremental_highlight import _fold_bare_whitespace_runs, _font

    if not isinstance(new_text, str):
        raise TypeError("new_text must be a string")
    spans = [(int(start), int(end)) for start, end in blue_spans]
    if any(start < 0 or start >= end or end > len(new_text) for start, end in spans):
        raise ValueError("Blue span outside rewritten text")
    if not new_text:
        return ""

    default = _font(base_font, None)
    default.color = copy.copy(base_font.color) if base_font is not None else None
    before = text(value)
    old_fonts: list[InlineFont] = []
    if isinstance(value, CellRichText):
        for part in value:
            segment = part.text or "" if isinstance(part, TextBlock) else str(part)
            font = part.font if isinstance(part, TextBlock) else default
            old_fonts.extend(copy.copy(font) for _ in segment)
    else:
        old_fonts = [copy.copy(default) for _ in before]
    if len(old_fonts) != len(before):
        raise ValueError("Original rich-text runs do not cover displayed text")

    new_fonts: list[InlineFont | None] = [None] * len(new_text)
    for opcode, left, right, new_left, new_right in SequenceMatcher(
            None, before, new_text, autojunk=False).get_opcodes():
        if opcode == "equal":
            for offset, font in enumerate(old_fonts[left:right]):
                new_fonts[new_left + offset] = copy.copy(font)
        elif opcode in {"replace", "insert"}:
            inherited = old_fonts[left] if left < len(old_fonts) else (
                old_fonts[left - 1] if left else default)
            for index in range(new_left, new_right):
                new_fonts[index] = copy.copy(inherited)
    if any(font is None for font in new_fonts):
        raise AssertionError("Rich-text rewrite left an uncovered character")

    for index, font in enumerate(new_fonts):
        assert font is not None
        is_blue = any(start <= index < end for start, end in spans)
        if is_blue:
            font.color = blue_color
        elif reset_existing_blue and font.color is not None and font.color.type == "rgb" and (
                str(font.color.rgb).upper().endswith(("0432FF", "0000FF"))):
            font.color = copy.copy(default.color)

    output = CellRichText()
    start = 0
    for index in range(1, len(new_text) + 1):
        if index == len(new_text) or new_fonts[index] != new_fonts[start]:
            output.append(TextBlock(text=new_text[start:index], font=copy.copy(new_fonts[start])))
            start = index
    output = _fold_bare_whitespace_runs(output)
    if str(output) != new_text:
        raise ValueError("Rich-text rewrite changed displayed text")
    return output


def rich_value_signature(value: Any, base_font: Any = None) -> dict[str, Any]:
    """Return a stable signature for displayed text and rich-text run formatting."""
    if base_font is not None:
        value = canonical_rich_value(value, base_font)
    displayed = text(value)
    return {
        "kind": "rich" if isinstance(value, CellRichText) else "plain",
        "text_sha256": hashlib.sha256(displayed.encode("utf-8")).hexdigest(),
        "runs": _rich_signature(value),
    }


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
    target_row: int | None = None,
) -> None:
    """Copy row geometry and cell styles without copying cell values."""
    destination_row = row if target_row is None else target_row
    source_dimension = source_sheet.row_dimensions.get(row)
    if source_dimension is None:
        if destination_row in target_sheet.row_dimensions:
            del target_sheet.row_dimensions[destination_row]
    else:
        target_dimension = target_sheet.row_dimensions[destination_row]
        for attribute in (
            "height", "hidden", "outlineLevel", "collapsed", "thickTop", "thickBot",
        ):
            setattr(target_dimension, attribute, copy.copy(getattr(source_dimension, attribute)))
    selected = columns if columns is not None else range(1, source_sheet.max_column + 1)
    for column in selected:
        copy_cell_style(source_sheet.cell(row, column), target_sheet.cell(destination_row, column))


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
            rich = _rich_signature(canonical_rich_value(cell.value, cell.font))
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


def delete_cols_with_manifest(sheet, start: int, amount: int) -> dict[str, Any]:
    """Delete physical columns with audit; reject dependencies we cannot rebase safely.

    Formula/chart/table/validation dependencies are intentionally not rewritten.
    Callers must first resolve these in a separate approved structural plan.
    """
    from openpyxl.utils import get_column_letter, column_index_from_string
    if start < 1 or amount < 0:
        raise ValueError("Invalid column deletion range")
    if amount == 0:
        return {"columns": [], "values": {}, "annotations": {}}
    if (sheet._charts or sheet._images or sheet.tables or sheet.data_validations.count
            or sheet.conditional_formatting or sheet.freeze_panes or sheet.auto_filter.ref
            or sheet.parent.defined_names
            or any(ws._charts or ws.tables or ws.data_validations.count or ws.conditional_formatting
                   or ws.defined_names for ws in sheet.parent)
            or any(cell.data_type == "f" for ws in sheet.parent for cell in ws._cells.values())):
        raise ValueError("Column deletion requires an explicit dependency-rebasing plan")
    end = start + amount - 1
    merges = []
    for merged in sheet.merged_cells.ranges:
        if merged.min_col < start <= merged.max_col or merged.min_col <= end < merged.max_col:
            raise ValueError("Merged range crosses deletion boundary")
        if merged.max_col < start:
            merges.append(str(merged))
        elif merged.min_col > end:
            merges.append(f"{get_column_letter(merged.min_col-amount)}{merged.min_row}:"
                          f"{get_column_letter(merged.max_col-amount)}{merged.max_row}")
    dimensions = []
    for key, dimension in sheet.column_dimensions.items():
        lo = dimension.min or column_index_from_string(key)
        hi = dimension.max or lo
        if lo < start <= hi or lo <= end < hi:
            raise ValueError("Column dimension group crosses deletion boundary")
        dimensions.append((lo, hi, copy.copy(dimension)))
    removed = [cell for (row, col), cell in sheet._cells.items() if start <= col <= end]
    manifest = {
        "columns": list(range(start, end+1)),
        "values": {cell.coordinate: text(cell.value) for cell in removed if cell.value is not None},
        "styles": {cell.coordinate: _style_signature(cell) for cell in removed if cell.has_style},
        "annotations": {cell.coordinate: {"text": cell.comment.text, "author": cell.comment.author}
                        for cell in removed if cell.comment},
        "column_dimensions": {str(lo): _dimension_signature(dim) for lo, hi, dim in dimensions
                              if start <= lo <= end},
        "merged_ranges": {"before": sorted(str(x) for x in sheet.merged_cells.ranges), "after": sorted(merges)},
    }
    # Remove merge metadata before moving cells, so unmerge cannot delete relocated cells.
    for merged in list(sheet.merged_cells.ranges):
        sheet.unmerge_cells(str(merged))
    sheet.delete_cols(start, amount)
    sheet.column_dimensions.clear()
    for lo, hi, dimension in dimensions:
        if start <= lo <= end:
            continue
        shift = amount if lo > end else 0
        dimension.min, dimension.max = lo-shift, hi-shift
        dimension.index = get_column_letter(lo-shift)
        sheet.column_dimensions[dimension.index] = dimension
    for merged in merges:
        sheet.merge_cells(merged)
    return manifest


def prune_right_columns_with_manifest(sheet, last_column: int) -> dict[str, Any]:
    """Remove a review-only right suffix without rebasing retained coordinates.

    This is for delivery templates whose A:D content is retained and all E+
    review material is discarded. Formulas in the discarded suffix may be
    dropped; retained formulas must refer only to retained columns. Unsupported
    workbook objects are rejected before mutation.
    """
    from openpyxl.utils import column_index_from_string

    if last_column < 1:
        raise ValueError("last_column must be positive")
    book = sheet.parent
    if (book.defined_names or sheet.defined_names or sheet._charts or sheet._images
            or sheet.tables or sheet.data_validations.count or sheet.conditional_formatting
            or sheet.freeze_panes or sheet.auto_filter.ref or sheet.print_area):
        raise ValueError("Right-column pruning requires an explicit dependency plan")
    for merged in sheet.merged_cells.ranges:
        if merged.min_col <= last_column < merged.max_col:
            raise ValueError("Merged range crosses pruning boundary")
    for cell in sheet._cells.values():
        if cell.column > last_column or cell.data_type != "f":
            continue
        formula = str(cell.value or "").upper()
        if any(token in formula for token in ("INDIRECT(", "OFFSET(", "ADDRESS(")):
            raise ValueError("Retained formula has a dynamic reference")
        for column in re.findall(r"(?<![A-Z0-9_])\$?([A-Z]{1,3})\$?\d+", formula):
            if column_index_from_string(column) > last_column:
                raise ValueError(f"Retained formula references removed column: {cell.coordinate}")
    for merged in list(sheet.merged_cells.ranges):
        if merged.min_col > last_column:
            sheet.unmerge_cells(str(merged))
    removed = [cell for cell in sheet._cells.values() if cell.column > last_column]
    manifest = {
        "removed_values": {cell.coordinate: text(cell.value) for cell in removed if cell.value is not None},
        "removed_styles": [cell.coordinate for cell in removed if cell.has_style],
        "removed_comments": [cell.coordinate for cell in removed if cell.comment],
        "removed_dimensions": [],
        "clipped_dimensions": [],
    }
    sheet._cells = {key: cell for key, cell in sheet._cells.items() if key[1] <= last_column}
    for key, dim in list(sheet.column_dimensions.items()):
        lo = dim.min or column_index_from_string(key)
        hi = dim.max or lo
        if lo > last_column:
            manifest["removed_dimensions"].append(key)
            del sheet.column_dimensions[key]
        elif hi > last_column:
            manifest["clipped_dimensions"].append({"key": key, "before_max": hi, "after_max": last_column})
            dim.max = last_column
    return manifest


class WorkbookTextSafetyError(ValueError):
    """A non-retryable failure: never publish the candidate workbook."""

    code = "workbook_text_safety_failed"


def canonical_rich_value(value: Any, base_font: Any) -> Any:
    """Canonical, Excel-safe encoding; no visible character is recoloured.

    Whitespace-only runs absorb into neighbours using the existing shared fix.
    Plain fragments get an explicit cell font so Excel cannot substitute its
    default font after a coloured run. No glossary/highlight renderer is run.
    """
    if not isinstance(value, CellRichText):
        return value
    from workbook_incremental_highlight import _fold_bare_whitespace_runs, _font

    default = _font(base_font, None)
    default.color = copy.copy(base_font.color)
    original_text = text(value)
    original_fonts = [signature for part in value
                      for signature in [str(part.font if isinstance(part, TextBlock) else default)] * len(text(part))]
    folded = _fold_bare_whitespace_runs(value)
    result = CellRichText()
    for part in folded:
        segment = text(part)
        if not segment:
            continue
        font = part.font if isinstance(part, TextBlock) else default
        result.append(TextBlock(font=copy.copy(font), text=segment))
    if text(result) != original_text:
        raise WorkbookTextSafetyError("Whitespace protection changed displayed text")
    protected_fonts = [signature for part in result for signature in [str(part.font)] * len(part.text)]
    if any(original_fonts[index] != protected_fonts[index] for index, character in enumerate(original_text)
           if not character.isspace()):
        raise WorkbookTextSafetyError("Whitespace protection changed a visible character's font/colour")
    return result


def prepare_workbook_for_save(workbook) -> list[str]:
    """Mandatory whole-workbook protection, including untouched/hidden sheets."""
    if not isinstance(workbook, openpyxl.Workbook) or workbook.write_only or workbook.read_only:
        raise WorkbookTextSafetyError("A normal inspectable Workbook is required; opaque writers are forbidden")
    changed = []
    for sheet in workbook:
        for cell in list(sheet._cells.values()):
            if isinstance(cell.value, CellRichText):
                safe = canonical_rich_value(cell.value, cell.font)
                if rich_value_signature(safe) != rich_value_signature(cell.value):
                    cell.value = safe
                    changed.append(f"{sheet.title}!{cell.coordinate}")
    return changed


def _text_font_snapshot(workbook) -> dict:
    """Exact text (spaces/CR/LF included) and character-font coverage."""
    from workbook_incremental_highlight import _font
    result = {}
    for sheet in workbook:
        for cell in sheet._cells.values():
            if not isinstance(cell.value, (str, CellRichText)) or not text(cell.value):
                continue
            value = canonical_rich_value(cell.value, cell.font)
            if not isinstance(value, CellRichText):
                font = _font(cell.font, None)
                font.color = copy.copy(cell.font.color)
                value = CellRichText(TextBlock(font=font, text=text(value)))
            # Segmentation alone is immaterial, but every character's font is not.
            runs = []
            for part in value:
                signature = str(part.font)
                if runs and runs[-1][1] == signature:
                    runs[-1][0] += part.text
                else:
                    runs.append([part.text, signature])
            result[(sheet.title, cell.coordinate)] = {"text": text(value), "fonts": runs}
    return result


def _preserve_serialized_whitespace(path: Path) -> None:
    """Set xml:space on text nodes, including wholly blank cells/runs.

    openpyxl omits it on whitespace-only text. Patch only opening t tags in
    worksheet/shared-string parts; all other ZIP members stay byte-identical.
    """
    tag = re.compile(rb'<(?:[A-Za-z_][\w.-]*:)?t(?=[\s>])[^>]*>')
    def preserve(match):
        value = match[0]
        if b'xml:space=' in value:
            return re.sub(rb'xml:space=[\"\'][^\"\']*[\"\']', b'xml:space="preserve"', value)
        return value[:-1] + b' xml:space="preserve">'
    patched = path.with_name(path.name + ".xmlspace.tmp")
    try:
        with ZipFile(path) as source, ZipFile(patched, "w") as target:
            target.comment = source.comment
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename == "xl/sharedStrings.xml" or (
                    info.filename.startswith("xl/worksheets/") and info.filename.endswith(".xml")
                ):
                    data = tag.sub(preserve, data)
                    # XML parsers normalize literal CR/CRLF to LF, including
                    # inside rich-text t nodes. Character references retain CR.
                    data = re.sub(rb'(<(?:[A-Za-z_][\w.-]*:)?t(?=[\s>])[^>]*>)(.*?)(</(?:[A-Za-z_][\w.-]*:)?t>)',
                                  lambda m: m[1] + m[2].replace(b'\r', b'&#13;') + m[3], data, flags=re.S)
                target.writestr(info, data)
        os.replace(patched, path)
    finally:
        patched.unlink(missing_ok=True)


def assert_serialized_text_safety(path: str | Path) -> dict:
    """Read-only gate for *any* candidate, also called before delivery/resume."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    preserve = "{http://www.w3.org/XML/1998/namespace}space"
    checked = 0
    with ZipFile(path) as archive:
        for name in archive.namelist():
            if name != "xl/sharedStrings.xml" and not (
                name.startswith("xl/worksheets/") and name.endswith(".xml")
            ):
                continue
            data = archive.read(name)
            if re.search(rb'<(?:[A-Za-z_][\w.-]*:)?t(?=[\s>])[^>]*>[^<]*\r', data):
                raise WorkbookTextSafetyError(f"Unescaped carriage return in {name}")
            root = ET.fromstring(data)
            for node in root.iter(ns + "t"):
                value = node.text or ""
                checked += 1
                if value and (value[0].isspace() or value[-1].isspace()) and node.get(preserve) != "preserve":
                    raise WorkbookTextSafetyError(f"Unprotected whitespace/line break in {name}")
            for container in [*root.iter(ns + "is"), *root.iter(ns + "si")]:
                runs = container.findall(ns + "r")
                displayed = "".join(node.text or "" for node in container.iter(ns + "t"))
                for run in runs:
                    node = run.find(ns + "t")
                    value = node.text or "" if node is not None else ""
                    if displayed.strip() and value and not value.strip():
                        raise WorkbookTextSafetyError(f"Whitespace-only rich-text run in {name}")
                    if value and run.find(ns + "rPr") is None:
                        raise WorkbookTextSafetyError(f"Implicit rich-text font in {name}")
    return {"status": "passed", "version": 1, "text_nodes_checked": checked}


def verify_text_font_roundtrip(path: str | Path, expected: dict) -> dict:
    safety = assert_serialized_text_safety(path)
    book = openpyxl.load_workbook(path, rich_text=True, data_only=False)
    try:
        actual = _text_font_snapshot(book)
    finally:
        book.close()
    differences = [f"{sheet}!{coordinate}" for sheet, coordinate in sorted(set(expected) | set(actual))
                   if expected.get((sheet, coordinate)) != actual.get((sheet, coordinate))]
    if differences:
        raise WorkbookTextSafetyError("Space/line-break/character-font mismatch: " + ", ".join(differences[:20]))
    return {**safety, "checks": {key: "passed" for key in ("whitespace", "linebreaks", "character_fonts")}}


def detailed_workbook_snapshot(workbook) -> dict[str, dict[str, Any]]:
    """Property-addressable five-axis facts, shared by independent disk verification.

    XML-bearing objects use full semantic XML rather than counts or style-table IDs.
    Binary drawing payloads and unsupported OOXML parts are handled by the disk verifier.
    """
    from openpyxl.xml.functions import tostring
    axes = {name: {} for name in ("values", "layout", "styles", "rich_text", "annotations")}
    def put(axis, path, value):
        axes[axis][json.dumps(path, ensure_ascii=False, separators=(",", ":"))] = value
    def xml(obj):
        return tostring(obj.to_tree()).decode() if obj is not None else None
    put("layout", ["sheetnames"], workbook.sheetnames)
    put("layout", ["defined_names"], sorted(xml(x) for x in workbook.defined_names.values()))
    put("layout", ["calculation"], xml(workbook.calculation))
    for sheet in workbook:
        base = ["sheets", sheet.title]
        for key, value in {
            "state": sheet.sheet_state, "max_row": sheet.max_row, "max_column": sheet.max_column,
            "merged_ranges": sorted(str(x) for x in sheet.merged_cells.ranges),
            "freeze_panes": str(sheet.freeze_panes) if sheet.freeze_panes else None,
            "sheet_properties": xml(sheet.sheet_properties), "sheet_format": xml(sheet.sheet_format),
            "protection": xml(sheet.protection), "auto_filter": xml(sheet.auto_filter),
            "views": xml(sheet.views), "page_margins": xml(sheet.page_margins),
            "header_footer": xml(sheet.HeaderFooter),
            "defined_names": sorted(xml(x) for x in sheet.defined_names.values()),
            "page_setup": xml(sheet.page_setup), "print_options": xml(sheet.print_options),
            "print_area": str(sheet.print_area), "print_title_rows": sheet.print_title_rows,
            "print_title_cols": sheet.print_title_cols, "row_breaks": xml(sheet.row_breaks),
            "col_breaks": xml(sheet.col_breaks),
            "data_validations": xml(sheet.data_validations),
            "tables": sorted(xml(t) for t in sheet.tables.values()),
            "conditional_formatting": sorted((str(cf.sqref), [xml(rule) for rule in rules])
                                             for cf, rules in sheet.conditional_formatting._cf_rules.items()),
        }.items():
            put("layout", base + [key], value)
        for axis_name, dimensions in (("row_dimensions", sheet.row_dimensions), ("column_dimensions", sheet.column_dimensions)):
            for index, dimension in sorted(dimensions.items()):
                path = base + [axis_name, str(index)]
                for key, value in _dimension_signature(dimension).items():
                    put("layout", path + [key], value)
                for key in ("min", "max", "bestFit"):
                    if hasattr(dimension, key):
                        put("layout", path + [key], getattr(dimension, key))
                if dimension.has_style:
                    for key, value in _style_signature(dimension).items():
                        put("styles", path + [key], value)
        for (row, col), cell in sorted(sheet._cells.items()):
            path = base + ["cells", cell.coordinate]
            if cell.value is not None:
                put("values", path, {"text": text(cell.value), "data_type": cell.data_type})
            if cell.has_style:
                for key, value in _style_signature(cell).items():
                    put("styles", path + [key], value)
            # Compare safe encodings of both authorities and candidates. The
            # disk gate separately rejects unsafe actual XML, so normalizing
            # run boundaries cannot conceal a missing space or changed font.
            rich = _rich_signature(canonical_rich_value(cell.value, cell.font))
            if rich is not None:
                put("rich_text", path, rich)
            if cell.comment:
                put("annotations", path + ["comment"], {"text": cell.comment.text, "author": cell.comment.author})
            if cell.hyperlink:
                put("annotations", path + ["hyperlink"], {key: getattr(cell.hyperlink, key)
                    for key in ("target", "location", "tooltip", "display")})
    return axes


def save_verified_atomic(
    workbook,
    output: str | Path,
    verify_path: Callable[[Path], Any] | None = None,
    *,
    overwrite: bool = False,
    contract: dict | None = None,
) -> Any:
    """Protect, serialize, independently verify text/fonts, then atomically save.

    Custom verifiers are additive; they cannot disable the mandatory safety
    gate. Contract executions additionally require independent five-axis proof.
    """
    destination = Path(output)
    if contract is not None:
        from workbook_contract import validate_contract
        validate_contract(contract)
        root = Path(contract["staging_root"]).resolve()
        if not destination.resolve().is_relative_to(root) or destination.resolve() == root:
            raise ValueError("Contract saves are restricted to staging")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {destination}")
    prepare_workbook_for_save(workbook)
    expected_text_fonts = _text_font_snapshot(workbook)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.verify-",
        suffix=".xlsx",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        token = _external_saves.set(False)
        try:
            workbook.save(temporary)
        finally:
            _external_saves.reset(token)
        _preserve_serialized_whitespace(temporary)
        safety = verify_text_font_roundtrip(temporary, expected_text_fonts)
        saved_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
        result = verify_path(temporary) if verify_path is not None else {"status": "passed"}
        if result is not None and (not isinstance(result, dict) or result.get("status") not in {"passed", "verified", "ok"}):
            raise WorkbookTextSafetyError("Additional workbook verification did not pass")
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != saved_hash:
            raise WorkbookTextSafetyError("Candidate changed during verification")
        if contract is not None:
            from workbook_contract import file_sha256, check_inputs, digest
            if (not isinstance(result, dict) or result.get("status") != "passed"
                    or result.get("staged_file_sha256") != file_sha256(temporary)
                    or result.get("contract_content_sha256") != digest(contract)
                    or set(result.get("checks", {})) != {"values", "layout", "styles", "rich_text", "annotations"}
                    or any(value != "passed" for value in result["checks"].values())):
                raise ValueError("Contract save requires a complete verification record")
            # A custom callback cannot certify its own output. Re-run the
            # independent verifier against the persisted approved contract.
            from workbook_verifier import verify_artifact
            persisted = Path(contract["work_root"]) / "contract.json"
            if not persisted.is_file() or json.loads(persisted.read_text()) != contract:
                raise ValueError("Contract save requires a persisted matching contract record")
            independent = verify_artifact(persisted, result.get("artifact_id"), temporary)
            if any(result.get(key) != independent[key] for key in (
                "artifact_id", "axis_sha256", "verifier_version", "text_safety",
                "contract_sha256", "input_authority_sha256",
            )):
                raise ValueError("Contract save requires an independent verification record")
            check_inputs(contract)
        if isinstance(result, dict) and contract is None:
            result = {**result, "text_safety": safety}
        os.replace(temporary, destination)
        return result
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def save_workbook_atomic(workbook, output: str | Path, *, overwrite: bool = False) -> Any:
    """Legacy/local writer entry: mandatory protection + text/font read-back.

    Approved edits/deliveries use the contract runner for authority comparisons.
    There is deliberately no flag for skipping protection or verification.
    """
    return save_verified_atomic(workbook, output, overwrite=overwrite)
