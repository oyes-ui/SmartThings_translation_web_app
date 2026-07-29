from __future__ import annotations

import sys
import unittest
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from workbook_review_apply import (  # noqa: E402
    _apply_decisions,
    _parse_c_range,
    _parse_columns,
    _snapshot_protected,
    _validate_decisions,
    _verify_protected,
)


class WorkbookReviewApplyTests(unittest.TestCase):
    def _workbook(self):
        wb = openpyxl.Workbook()
        target = wb.active
        target.title = "FR(프랑스)"
        target["C8"] = "current"
        target["F8"] = "reviewer"
        target["H8"] = "native comment"
        target["C10"] = "current partial"
        target["F10"] = "ignored full proposal"
        target["C11"] = "keep"
        protected = wb.create_sheet("US(미국)")
        protected["C8"] = "source"
        protected["I3"] = "must remain"
        protected.merge_cells("A1:B1")
        return wb

    def test_accept_partial_hold_apply_only_approved_c_values(self):
        wb = self._workbook()
        decisions = _validate_decisions(
            wb,
            [
                {"sheet": "FR(프랑스)", "cell": "C8", "decision": "accept", "source_sheet": "US(미국)"},
                {"sheet": "FR(프랑스)", "cell": "C10", "decision": "partial", "final_value": "approved partial"},
                {"sheet": "FR(프랑스)", "cell": "C11", "decision": "hold"},
            ],
            {"US(미국)"},
            "C7:C28",
        )
        records, changes = _apply_decisions(wb, decisions)
        self.assertEqual(wb["FR(프랑스)"]["C8"].value, "reviewer")
        self.assertEqual(wb["FR(프랑스)"]["C10"].value, "approved partial")
        self.assertEqual(wb["FR(프랑스)"]["C11"].value, "keep")
        self.assertEqual([entry["decision"] for entry in records], ["accept", "partial", "hold"])
        self.assertEqual([entry["decision"] for entry in changes], ["accept", "partial"])

    def test_protected_sheet_and_wrong_source_are_rejected(self):
        wb = self._workbook()
        with self.assertRaises(ValueError):
            _validate_decisions(
                wb, [{"sheet": "US(미국)", "cell": "C8", "decision": "accept"}], {"US(미국)"}, "C7:C28"
            )
        with self.assertRaises(ValueError):
            _validate_decisions(
                wb,
                [{"sheet": "FR(프랑스)", "cell": "C8", "decision": "accept", "source_sheet": "KR(한국)"}],
                {"US(미국)"}, "C7:C28"
            )

    def test_protected_snapshot_allows_explicit_review_column_removal_only(self):
        wb = self._workbook()
        before = _snapshot_protected(wb, {"US(미국)"}, (5, 8))
        for ws in wb.worksheets:
            ws.delete_cols(5, 4)
        after = _snapshot_protected(wb, {"US(미국)"})
        self.assertTrue(_verify_protected(before, after)["passed"])
        wb["US(미국)"]["C8"] = "changed"
        with self.assertRaises(RuntimeError):
            _verify_protected(before, _snapshot_protected(wb, {"US(미국)"}))

    def test_only_c_range_and_column_range_are_accepted(self):
        self.assertEqual(_parse_c_range("C7:C28"), (7, 28))
        self.assertEqual(_parse_columns("E:H"), (5, 8))
        with self.assertRaises(ValueError):
            _parse_c_range("B7:C28")


if __name__ == "__main__":
    unittest.main()
