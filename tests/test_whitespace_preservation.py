# -*- coding: utf-8 -*-
"""
Whitespace-preservation regression tests for the highlight / save pipeline.

Locks in the 1st-round fixes for the "missing space" issue
(e.g. ``SmartThings Family Care`` -> ``SmartThingsFamily Care``):

  1. highlight-only reads the cell WITHOUT ``strip()`` (``_cell_text_for_highlighting``)
  2. the rich-text writer (``_apply_rich_text``) reproduces every character, including
     leading / trailing / interior / NBSP whitespace, and openpyxl round-trips it intact
  3. the post-save guard (``_verify_saved_cell_text``) fails closed on any drift
  4. the deterministic bracket strip never joins two adjacent words in well-formed input

These paths carry no LLM cost, so the tests run offline.
"""
import os
import re
import sys
import tempfile
import unittest
import zipfile

import openpyxl
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont

from translation_web_app.checker_service import TranslationChecker

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Strings that specifically stress the reported failure mode: a brand phrase whose
# interior space must survive, plus edge whitespace the old ``strip()`` used to eat.
_WHITESPACE_CASES = [
    ("SmartThings Family Care", ["SmartThings", "Family Care"]),
    ("Galaxy Device control is available.", ["Galaxy Device", "control"]),
    ("  leading spaces kept", ["leading"]),
    ("trailing spaces kept   ", ["trailing"]),
    ("double  space  inside", ["double"]),
    ("SmartThings Family Care", ["SmartThings"]),   # NBSP between brand words
    ("여러 단어 SmartThings Family Care 포함", ["SmartThings", "Family Care"]),
]


class CellTextForHighlightingTests(unittest.TestCase):
    def setUp(self):
        self.checker = TranslationChecker()

    def test_does_not_strip_edges(self):
        self.assertEqual(
            self.checker._cell_text_for_highlighting("  SmartThings Family Care  "),
            "  SmartThings Family Care  ",
        )

    def test_none_becomes_empty_string(self):
        self.assertEqual(self.checker._cell_text_for_highlighting(None), "")

    def test_regression_old_strip_would_have_dropped_edges(self):
        # Guards against anyone reintroducing the ``.strip()`` that caused the bug.
        raw = "  SmartThings Family Care  "
        self.assertNotEqual(self.checker._cell_text_for_highlighting(raw), raw.strip())


class RichTextRoundTripTests(unittest.TestCase):
    """_apply_rich_text -> save -> reload must preserve the full character stream."""

    def setUp(self):
        self.checker = TranslationChecker()

    def _round_trip(self, text, keywords):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "S"
        ws["A1"].value = self.checker._apply_rich_text(text, keywords, base_font=ws["A1"].font)
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as fh:
            path = fh.name
        try:
            wb.save(path)
            reloaded = openpyxl.load_workbook(path, data_only=False, rich_text=True)
            value = reloaded["S"]["A1"].value
            return "" if value is None else str(value)
        finally:
            os.unlink(path)

    def test_all_whitespace_shapes_survive(self):
        for text, keywords in _WHITESPACE_CASES:
            with self.subTest(text=text):
                self.assertEqual(self._round_trip(text, keywords), text)

    def test_no_keyword_match_preserves_text(self):
        # No highlight applied -> plain string returned unchanged.
        self.assertEqual(self._round_trip("SmartThings Family Care", ["NoSuchTerm"]),
                         "SmartThings Family Care")


class RichTextNoBareSpaceRunTests(unittest.TestCase):
    """A rich-text run that is ONLY whitespace ships without xml:space="preserve"
    (openpyxl's ``whitespace()`` requires non-empty stripped text), so Excel trims it and
    glues consecutive highlighted glossary terms ("Galaxy Device" -> "GalaxyDevice").
    openpyxl's own reader does NOT trim, so this must be checked at the XML level.
    """

    def setUp(self):
        self.checker = TranslationChecker()

    _T_RUN = re.compile(r'<t(\s[^>]*)?>(.*?)</t>', re.DOTALL)

    def _bare_space_runs(self, text, keywords):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "S"
        ws["A1"].value = self.checker._apply_rich_text(text, keywords, base_font=ws["A1"].font)
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as fh:
            path = fh.name
        try:
            wb.save(path)
            bad = []
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.startswith("xl/worksheets/") and name.endswith(".xml"):
                        xml = z.read(name).decode("utf-8")
                        for attr, content in self._T_RUN.findall(xml):
                            # a run that is only whitespace WITHOUT an xml:space="preserve"
                            if content and content.strip() == "" and "preserve" not in (attr or ""):
                                bad.append(content)
            return bad
        finally:
            os.unlink(path)

    def test_consecutive_terms_emit_no_bare_space_run(self):
        cases = [
            ("Galaxy Device control", ["Galaxy", "Device"]),
            ("SmartThings Family Care", ["SmartThings", "Family Care"]),
            ("Bixby Galaxy SmartThings", ["Bixby", "Galaxy", "SmartThings"]),   # 3 consecutive
        ]
        for text, keywords in cases:
            with self.subTest(text=text):
                self.assertEqual(self._bare_space_runs(text, keywords), [])

    def test_space_between_consecutive_terms_survives_strict_reload(self):
        # Simulate a strict reader by re-parsing the <t> runs and dropping any bare space
        # run (what Excel does); the reconstructed text must still contain the space.
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "S"
        ws["A1"].value = self.checker._apply_rich_text("Galaxy Device", ["Galaxy", "Device"],
                                                       base_font=ws["A1"].font)
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as fh:
            path = fh.name
        try:
            wb.save(path)
            with zipfile.ZipFile(path) as z:
                xml = next(z.read(n).decode("utf-8") for n in z.namelist()
                           if n.startswith("xl/worksheets/") and n.endswith(".xml"))
            reconstructed = "".join(
                content for attr, content in self._T_RUN.findall(xml)
                if not (content and content.strip() == "" and "preserve" not in (attr or ""))
            )
            self.assertIn("Galaxy Device", reconstructed)
        finally:
            os.unlink(path)


class WorkbookWhitespaceHardeningTests(unittest.TestCase):
    """A bare space run can also enter a workbook from an Excel round-trip or an earlier
    pipeline pass (not just our own writer). The pre-save hardening pass must fold any
    such run — otherwise Excel trims it, dropping the space AND bleeding the neighbouring
    run's colour across the merge ("Bixby ผู้ช่วย" -> blue "Bixbyผู้ช่วย...").
    """

    _T_RUN = re.compile(r'<t(\s[^>]*)?>(.*?)</t>', re.DOTALL)

    def setUp(self):
        self.checker = TranslationChecker()

    def _fragile_cell(self):
        # blue term, then a STANDALONE space run, then following text (the shape Excel
        # leaves after re-segmenting a highlighted cell).
        return CellRichText([
            TextBlock(InlineFont(), "พบกับ "),
            TextBlock(InlineFont(color="0000FF"), "Bixby"),
            TextBlock(InlineFont(), " "),               # <-- bare whitespace run
            TextBlock(InlineFont(), "ผู้ช่วย"),
        ])

    def test_harden_cell_folds_bare_space_and_preserves_text(self):
        hardened = self.checker._harden_richtext_cell(self._fragile_cell())
        self.assertEqual(str(hardened), "พบกับ Bixby ผู้ช่วย")
        bare = [b for b in hardened
                if isinstance(b, TextBlock) and b.text and b.text.strip() == ""]
        self.assertEqual(bare, [])

    def test_harden_cell_leaves_clean_cell_identical(self):
        clean = CellRichText([
            TextBlock(InlineFont(color="0000FF"), "Bixby"),
            TextBlock(InlineFont(), " is ready"),
        ])
        self.assertIs(self.checker._harden_richtext_cell(clean), clean)

    def test_harden_workbook_emits_no_bare_space_run(self):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "S"
        ws["A1"].value = self._fragile_cell()
        self.assertEqual(self.checker._harden_workbook_whitespace_runs(wb), 1)
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as fh:
            path = fh.name
        try:
            wb.save(path)
            bad = []
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.endswith(".xml") and ("worksheets/" in name or "sharedStrings" in name):
                        xml = z.read(name).decode("utf-8")
                        for attr, content in self._T_RUN.findall(xml):
                            if content and content.strip() == "" and "preserve" not in (attr or ""):
                                bad.append(content)
            self.assertEqual(bad, [])
            # and the text survives a strict (Excel-style) reparse that drops bare runs
            self.assertEqual(str(openpyxl.load_workbook(path, rich_text=True)["S"]["A1"].value),
                             "พบกับ Bixby ผู้ช่วย")
        finally:
            os.unlink(path)


class VerifySavedCellTextGuardTests(unittest.TestCase):
    def setUp(self):
        self.checker = TranslationChecker()

    def _write(self, coord_to_text):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "S"
        for coord, text in coord_to_text.items():
            ws[coord].value = self.checker._apply_rich_text(text, ["SmartThings"], base_font=ws[coord].font)
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as fh:
            path = fh.name
        wb.save(path)
        return path

    def test_passes_when_text_matches(self):
        path = self._write({"A1": "SmartThings Family Care"})
        try:
            self.checker._verify_saved_cell_text(path, {("S", "A1"): "SmartThings Family Care"})
        finally:
            os.unlink(path)

    def test_raises_on_missing_interior_space(self):
        path = self._write({"A1": "SmartThings Family Care"})
        try:
            with self.assertRaises(RuntimeError):
                # The generator claims a space-less string; the saved cell has the space.
                self.checker._verify_saved_cell_text(path, {("S", "A1"): "SmartThingsFamily Care"})
        finally:
            os.unlink(path)


class BracketStripDoesNotJoinWordsTests(unittest.TestCase):
    """The only text-mutating guardrail (_unwrap_glossary_brackets) must never delete a
    space between two adjacent words in well-formed input."""

    def setUp(self):
        self.checker = TranslationChecker()

    def test_wrapper_removal_keeps_following_word_space(self):
        ctx = {"SmartThings": "SmartThings"}
        for wrapped in ("[SmartThings] Family Care", '"SmartThings" Family Care', "«SmartThings» Family Care"):
            with self.subTest(wrapped=wrapped):
                out = self.checker._strip_glossary_brackets_by_policy(wrapped, ctx, row_key="//sec_1_button")
                self.assertEqual(out, "SmartThings Family Care")

    def test_two_spaced_wrapped_terms_keep_their_gap(self):
        ctx = {"SmartThings": "SmartThings", "Family Care": "Family Care"}
        out = self.checker._strip_glossary_brackets_by_policy(
            "[SmartThings] [Family Care]", ctx, row_key="//sec_1_button"
        )
        self.assertEqual(out, "SmartThings Family Care")


class HighlightScriptTextPreservationGuardTests(unittest.TestCase):
    """The highlight CLI's own before/after equality guard (_verify_text_preservation)."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        script = os.path.join(
            _ROOT, "agent-packages", "smartthings-translation-agent",
            "scripts", "workbook_highlight_glossary.py",
        )
        if not os.path.exists(script):
            raise unittest.SkipTest("highlight script not present")
        spec = importlib.util.spec_from_file_location("_wh_guard", script)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - depends on _app_pipeline import
            raise unittest.SkipTest(f"highlight script not importable: {exc}")
        cls.verify = staticmethod(module._verify_text_preservation)

    def _wb(self, value):
        from pathlib import Path
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "S"
        ws["A1"].value = value
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as fh:
            path = fh.name
        wb.save(path)
        return Path(path)

    def test_identical_workbooks_pass(self):
        a = self._wb("SmartThings Family Care")
        b = self._wb("SmartThings Family Care")
        try:
            result = self.verify(a, b)
            self.assertEqual(result["changed_cells"], 0)
        finally:
            os.unlink(a)
            os.unlink(b)

    def test_changed_cell_fails_closed(self):
        a = self._wb("SmartThings Family Care")
        b = self._wb("SmartThingsFamily Care")
        try:
            with self.assertRaises(RuntimeError):
                self.verify(a, b)
        finally:
            os.unlink(a)
            os.unlink(b)


if __name__ == "__main__":
    unittest.main()
