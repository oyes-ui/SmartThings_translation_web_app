from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import openpyxl
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Font, PatternFill

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from workbook_mutation_guard import (  # noqa: E402
    clone_rich_value,
    copy_row_layout,
    delete_rows_with_manifest,
    insert_text_with_red,
    prune_right_columns_with_manifest,
    replace_characters_with_red,
    replace_span_with_red,
    save_verified_atomic,
    rich_value_signature,
    rewrite_rich_with_blue_spans,
    slice_rich_value,
    snapshot_axis_diff,
    snapshot_path,
)


class WorkbookMutationGuardTests(unittest.TestCase):
    def _book(self, path: Path, value: str = "source") -> Path:
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "CO(콜롬비아)"
        sheet["C7"] = value
        book.save(path)
        book.close()
        return path

    def test_clone_rich_value_keeps_independent_runs(self):
        value = CellRichText(
            TextBlock(InlineFont(color="0000FF"), "Smart"),
            TextBlock(InlineFont(b=True), "Things"),
        )
        cloned = clone_rich_value(value)
        self.assertEqual(str(cloned), "SmartThings")
        self.assertIsNot(cloned, value)
        self.assertEqual(str(cloned[0].font.color.rgb), str(value[0].font.color.rgb))
        self.assertIsNot(cloned[0].font, value[0].font)

    def test_slice_rich_value_keeps_overlapping_runs_and_fonts(self):
        value = CellRichText(
            TextBlock(InlineFont(color="0000FF"), "Prefix "),
            TextBlock(InlineFont(b=True), "Button"),
            " suffix",
        )
        sliced = slice_rich_value(value, 7, 13)
        self.assertEqual(str(sliced), "Button")
        self.assertEqual(len(sliced), 1)
        self.assertTrue(sliced[0].font.b)
        self.assertEqual(rich_value_signature(sliced)["kind"], "rich")

    def test_replace_span_with_red_removes_wrappers_and_preserves_other_runs(self):
        value = CellRichText(
            TextBlock(InlineFont(i=True), "Start "),
            TextBlock(InlineFont(color="0000FF", b=True), "[Pet Care]"),
            " today",
        )
        start = str(value).index("[Pet Care]")
        result = replace_span_with_red(value, start, start + len("[Pet Care]"), "Pet Care", Font(name="Arial"))
        self.assertEqual(str(result), "Start Pet Care today")
        self.assertEqual(result[0].font, value[0].font)
        self.assertEqual(str(result[-1]), " today")
        self.assertTrue(str(result[1].font.color.rgb).endswith("FF0000"))
        self.assertTrue(result[1].font.b)

    def test_replace_span_with_red_can_insert_or_delete(self):
        inserted = replace_span_with_red("Start Care", 6, 6, "Pet ", Font(name="Arial"))
        self.assertEqual(str(inserted), "Start Pet Care")
        self.assertTrue(str(inserted[1].font.color.rgb).endswith("FF0000"))
        deleted = replace_span_with_red("vc一张照片", 0, 2, "", Font(name="Arial"))
        self.assertEqual(str(deleted), "一张照片")

    def test_replace_characters_with_red_preserves_other_inline_fonts(self):
        value = CellRichText(
            TextBlock(InlineFont(color="0000FF", b=True), "SmartThings"),
            TextBlock(InlineFont(i=True, color="000000"), " don't stop"),
        )
        position = str(value).index("'")
        result = replace_characters_with_red(value, {position: "’"}, Font(name="Arial"))
        self.assertEqual(str(result), "SmartThings don’t stop")
        self.assertEqual(result[0].text, "SmartThings")
        self.assertEqual(result[0].font, value[0].font)
        self.assertEqual(result[1].text, " don")
        self.assertEqual(result[1].font, value[1].font)
        self.assertEqual(result[2].text, "’")
        self.assertTrue(str(result[2].font.color.rgb).endswith("FF0000"))
        self.assertTrue(result[2].font.i)
        self.assertEqual(result[3].font, value[1].font)

    def test_replace_characters_with_red_handles_plain_text(self):
        result = replace_characters_with_red("L'expérience", {1: "’"}, Font(name="Arial"))
        self.assertEqual(str(result), "L’expérience")
        self.assertTrue(str(result[1].font.color.rgb).endswith("FF0000"))

    def test_insert_text_with_red_preserves_existing_rich_fonts(self):
        value = CellRichText(
            TextBlock(InlineFont(color="0000FF", b=True), "First"),
            "\nSecond",
        )
        result = insert_text_with_red(
            value, {0: "* ", 6: "* ", len(str(value)): "\n* Final"}, Font(name="Arial")
        )
        self.assertEqual(str(result), "* First\n* Second\n* Final")
        first = next(part for part in result if isinstance(part, TextBlock) and part.text == "First")
        self.assertEqual(first.font, value[0].font)
        for part in result:
            if isinstance(part, TextBlock) and part.text.startswith("* "):
                self.assertEqual(str(part.font.color.rgb).upper(), "FFFF0000")

    def test_insert_text_with_red_rejects_invalid_offsets(self):
        with self.assertRaises(ValueError):
            insert_text_with_red("abc", {4: "x"}, Font())

    def test_rewrite_rich_keeps_noncolor_run_style_and_replaces_broad_blue(self):
        value = CellRichText(
            TextBlock(InlineFont(b=True, family=2, color="0432FF"), "Galaxy phone"),
            TextBlock(InlineFont(family=3, color="000000"), " or phone"),
        )
        result = rewrite_rich_with_blue_spans(
            value, "Galaxy phone or [Ring]", Font(name="Arial", color="000000"),
            [(0, 6), (17, 21)],
        )
        self.assertEqual(str(result), "Galaxy phone or [Ring]")
        chars = [(character, part.font) for part in result for character in part.text]
        self.assertTrue(chars[0][1].b)
        self.assertEqual(chars[0][1].family, 2)
        self.assertEqual(chars[7][1].family, 2)
        self.assertEqual(chars[13][1].family, 3)
        self.assertTrue(str(chars[0][1].color.rgb).endswith("0432FF"))
        self.assertFalse(str(chars[7][1].color.rgb).endswith("0432FF"))
        self.assertTrue(str(chars[17][1].color.rgb).endswith("0432FF"))

    def test_rewrite_rich_inserts_disclaimer_prefix_without_bracketing_noun(self):
        value = CellRichText(
            TextBlock(InlineFont(color="0432FF"), "Notify when left behind"),
            TextBlock(InlineFont(color="000000"), " uses a smartphone."),
        )
        target = "* Notify when left behind uses a smartphone."
        result = rewrite_rich_with_blue_spans(value, target, Font(color="000000"), [(2, 25)])
        self.assertEqual(str(result), target)
        self.assertIn("smartphone", str(result))
        self.assertNotIn("[smartphone]", str(result))

    def test_scoped_blue_highlight_preserves_existing_blue_and_red(self):
        value = CellRichText(
            TextBlock(InlineFont(color="0432FF", b=True), "Pet Care"),
            TextBlock(InlineFont(color="000000"), " > "),
            TextBlock(InlineFont(color="FF0000", i=True), "Life"),
        )
        result = rewrite_rich_with_blue_spans(
            value, str(value), Font(color="000000"), [(11, 15)],
            reset_existing_blue=False,
        )
        chars = [part.font for part in result for _ in part.text]
        self.assertTrue(str(chars[0].color.rgb).endswith("0432FF"))
        self.assertTrue(chars[0].b)
        self.assertTrue(str(chars[11].color.rgb).endswith("0432FF"))
        self.assertTrue(chars[11].i)

    def test_cross_workbook_row_layout_copies_semantics_not_style_ids(self):
        source = openpyxl.Workbook()
        target = openpyxl.Workbook()
        source.active.row_dimensions[7].height = 41
        source.active["C7"].font = Font(bold=True, color="00FF0000")
        source.active["C7"].fill = PatternFill("solid", fgColor="0000FF00")
        copy_row_layout(source.active, target.active, 7)
        self.assertEqual(target.active.row_dimensions[7].height, 41)
        self.assertTrue(target.active["C7"].font.bold)
        self.assertEqual(str(target.active["C7"].fill.fgColor.rgb), "0000FF00")
        copy_row_layout(source.active, target.active, 7, target_row=12)
        self.assertEqual(target.active.row_dimensions[12].height, 41)
        self.assertTrue(target.active["C12"].font.bold)
        source.close()
        target.close()

    def test_physical_row_delete_records_values_styles_dimensions_and_moves_following_rows(self):
        book = openpyxl.Workbook()
        sheet = book.active
        sheet["B15"] = "section"
        sheet["C15"] = "text"
        sheet["E15"] = "review note"
        sheet["C15"].fill = PatternFill("solid", fgColor="00FFFF00")
        sheet.row_dimensions[15].height = 90
        sheet["C16"] = "next"
        sheet.row_dimensions[16].height = 22
        deleted = delete_rows_with_manifest(sheet, 15, 1)
        self.assertEqual(deleted["values"]["E15"], "review note")
        self.assertIn("C15", deleted["styled_cells"])
        self.assertEqual(deleted["row_dimensions"]["15"]["height"], 90)
        self.assertEqual(sheet["C15"].value, "next")
        self.assertEqual(sheet.row_dimensions[15].height, 22)
        self.assertNotIn(16, sheet.row_dimensions)
        book.close()

    def test_snapshot_separates_value_style_and_layout_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._book(root / "source.xlsx")
            styled = root / "styled.xlsx"
            book = openpyxl.load_workbook(source)
            book.active["C7"].font = Font(bold=True)
            book.save(styled)
            book.close()
            style_diff = snapshot_axis_diff(snapshot_path(source), snapshot_path(styled))
            self.assertNotIn("values", style_diff)
            self.assertIn("styles", style_diff)

            structured = root / "structured.xlsx"
            book = openpyxl.load_workbook(styled)
            book.active["C15"] = "section"
            book.active.row_dimensions[15].height = 90
            book.save(structured)
            book.close()
            structure_diff = snapshot_axis_diff(snapshot_path(styled), snapshot_path(structured))
            self.assertIn("values", structure_diff)
            self.assertIn("layout", structure_diff)

    def test_failed_verification_never_replaces_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self._book(root / "output.xlsx", "old")
            before = output.read_bytes()
            book = openpyxl.Workbook()
            book.active["C7"] = "new"

            def fail(_path: Path):
                raise RuntimeError("verification failed")

            with self.assertRaises(RuntimeError):
                save_verified_atomic(book, output, fail, overwrite=True)
            self.assertEqual(output.read_bytes(), before)
            self.assertFalse(list(root.glob(".output.verify-*.xlsx")))

            def pass_check(path: Path):
                reopened = openpyxl.load_workbook(path)
                try:
                    self.assertEqual(reopened.active["C7"].value, "new")
                finally:
                    reopened.close()
                return {"status": "verified"}

            result = save_verified_atomic(book, output, pass_check, overwrite=True)
            book.close()
            self.assertEqual(result["status"], "verified")
            verified = openpyxl.load_workbook(output)
            self.assertEqual(verified.active["C7"].value, "new")
            verified.close()

    def test_prune_review_suffix_discards_only_right_columns(self):
        book = openpyxl.Workbook()
        sheet = book.active
        sheet["B7"] = '=RIGHT($C$5, 3)'
        sheet["C5"] = "story_001"
        sheet["C7"] = CellRichText(TextBlock(InlineFont(color="0000FF"), "SmartThings"))
        sheet["E7"] = '=RIGHT($C$5, 3)'
        sheet["F7"] = "review"
        sheet.column_dimensions["D"].min = 4
        sheet.column_dimensions["D"].max = 7
        sheet.column_dimensions["H"].min = 8
        sheet.column_dimensions["H"].max = 16384
        deleted = prune_right_columns_with_manifest(sheet, 4)
        self.assertEqual(deleted["removed_values"]["F7"], "review")
        self.assertEqual(sheet.max_column, 3)
        self.assertEqual(sheet["B7"].value, '=RIGHT($C$5, 3)')
        self.assertEqual(str(sheet["C7"].value), "SmartThings")
        self.assertEqual(sheet.column_dimensions["D"].max, 4)
        self.assertNotIn("H", sheet.column_dimensions)
        book.close()

    def test_prune_review_suffix_rejects_retained_formula_dependency(self):
        book = openpyxl.Workbook()
        sheet = book.active
        sheet["B7"] = '=E7'
        sheet["E7"] = "review"
        with self.assertRaisesRegex(ValueError, "references removed column"):
            prune_right_columns_with_manifest(sheet, 4)
        self.assertEqual(sheet["E7"].value, "review")
        book.close()


if __name__ == "__main__":
    unittest.main()
