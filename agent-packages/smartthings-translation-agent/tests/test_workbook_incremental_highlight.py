from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import openpyxl
from openpyxl.cell.rich_text import CellRichText, TextBlock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from workbook_incremental_highlight import _next_output_path, _term_spans, render_cell  # noqa: E402


def _runs(value):
    return [
        (part.text, part.font.color.rgb if isinstance(part, TextBlock) and part.font.color else None)
        for part in value
    ]


class IncrementalHighlightTests(unittest.TestCase):
    def test_blue_glossary_wins_over_red_change(self):
        value = render_cell(
            "Use SmartThings now", base_font=None,
            red_spans=[[4, 15]], blue_spans=[[4, 15]],
        )
        self.assertIsInstance(value, CellRichText)
        self.assertEqual(str(value), "Use SmartThings now")
        self.assertEqual(_runs(value), [("Use ", None), ("SmartThings", "000000FF"), (" now", None)])

    def test_non_overlapping_change_remains_red(self):
        value = render_cell(
            "Use SmartThings now", base_font=None,
            red_spans=[[16, 19]], blue_spans=[[4, 15]],
        )
        self.assertEqual(_runs(value), [("Use ", None), ("SmartThings", "000000FF"), (" ", None), ("now", "00FF0000")])

    def test_terms_are_case_insensitive_and_unmatched_are_reported(self):
        spans, missing = _term_spans("Use smartthings", ["SmartThings", "Galaxy"])
        self.assertEqual(spans, [[4, 15]])
        self.assertEqual(missing, ["Galaxy"])

    def test_output_never_reuses_existing_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            workbook = Path(tmp) / "story.xlsx"
            workbook.touch()
            first = _next_output_path(workbook, "20260803_120000")
            first.touch()
            second = _next_output_path(workbook, "20260803_120000")
            self.assertNotEqual(first, second)
            self.assertTrue(second.name.endswith("_2.xlsx"))

    def test_composed_runs_survive_xlsx_round_trip_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.xlsx"
            wb = openpyxl.Workbook()
            cell = wb.active["C7"]
            cell.value = render_cell(
                "Use SmartThings now", base_font=cell.font,
                red_spans=[[16, 19]], blue_spans=[[4, 15]],
            )
            wb.save(path)
            restored = openpyxl.load_workbook(path, rich_text=True)
            value = restored.active["C7"].value
            self.assertEqual(str(value), "Use SmartThings now")
            self.assertEqual(_runs(value), [("Use ", None), ("SmartThings", "000000FF"), (" ", None), ("now", "00FF0000")])


if __name__ == "__main__":
    unittest.main()
