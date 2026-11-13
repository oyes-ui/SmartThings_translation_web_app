from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from workbook_translate import validate_translation_output  # noqa: E402


class TranslationOutputValidationTests(unittest.TestCase):
    def _workbooks(self, directory: Path, target_value: str | None) -> tuple[Path, Path]:
        source = Workbook()
        source.active.title = "KR(한국)"
        source.create_sheet("CN(중국)")
        source["KR(한국)"]["C7"] = "번역할 문구"
        source["KR(한국)"]["C8"] = "x"
        source_path = directory / "source.xlsx"
        source.save(source_path)

        output = Workbook()
        output.active.title = "KR(한국)"
        output.create_sheet("CN(중국)")
        output["CN(중국)"]["C7"] = target_value
        output_path = directory / "output.xlsx"
        output.save(output_path)
        return source_path, output_path

    def test_rejects_error_text_in_saved_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = self._workbooks(Path(temp), "[번역 오류] Gemini Error")
            result = validate_translation_output(
                source, output,
                [{"source_sheet": "KR(한국)", "target_sheets": ["CN(중국)"]}],
                "KR(한국)", None, "C7:C8",
            )
            self.assertEqual(result["checked_cells"], 1)
            self.assertEqual(result["problem_count"], 1)
            self.assertIn("CN(중국)!C7", result["problems"][0])

    def test_accepts_translation_and_skips_x_placeholder(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = self._workbooks(Path(temp), "已翻译")
            result = validate_translation_output(
                source, output,
                [{"source_sheet": "KR(한국)", "target_sheets": ["CN(중국)"]}],
                "KR(한국)", None, "C7:C8",
            )
            self.assertEqual(result, {"checked_cells": 1, "problem_count": 0, "problems": []})


if __name__ == "__main__":
    unittest.main()
