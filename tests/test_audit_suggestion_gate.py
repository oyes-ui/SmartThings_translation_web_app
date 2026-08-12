# -*- coding: utf-8 -*-
"""Deterministic gate for model-proposed edits (GlossaryChecker.validate_audit_suggestion).

The translation path already repairs and re-validates model output before letting it
out. A reviewer proposing an edit had no equivalent gate, so a fluent suggestion that
replaced a settled glossary target shipped as a consensus proposal — ES_CO C18, where
three review roles proposed their own wording over `Movie mode → Modo de video`.

Exercised against the real shipped glossary rather than a fixture: the point is that
the resolver's own term data blocks the edit, so a synthetic term would prove nothing.
Skips automatically when the glossary is absent.
"""

import asyncio
import os
import unittest

from translation_web_app.glossary_checks import GlossaryChecker
from translation_web_app.prompt_builder import PromptBuilder

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOSSARY_CSV = os.path.join(_ROOT, "runtime", "glossary", "latest_glossary_260803.csv")

CO_CODE = "스페인어_콜롬비아"
SOURCE = "Watch in Movie mode."
RESOLVED = "Modo de video"


@unittest.skipUnless(os.path.exists(GLOSSARY_CSV), "shipped glossary not available")
class AuditSuggestionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = GlossaryChecker(PromptBuilder())
        loaded = asyncio.run(cls.checker.load_glossary_from_file(GLOSSARY_CSV, "영어_미국"))
        if not loaded.startswith("✓"):
            raise unittest.SkipTest(loaded)
        if "Movie mode" not in cls.checker.glossary:
            raise unittest.SkipTest("glossary no longer carries the C18 term")

    def _gate(self, suggestion, row_key="description"):
        card = self.checker.resolve_constraints(SOURCE, CO_CODE, row_key=row_key, cell="C18")
        context = self.checker._get_glossary_context_as_dict(
            CO_CODE, source_text=SOURCE, skip_deactivated=True, row_key=row_key)
        return self.checker.validate_audit_suggestion(
            suggestion, card, glossary_context=context, row_key=row_key)

    def test_the_c18_regression_a_competing_translation_is_blocked(self):
        """`Modo película` is the verbatim after value three ES_CO roles agreed on."""
        result = self._gate("Ve en Modo película.")
        self.assertTrue(result["blocked"])
        self.assertEqual(result["action"], "human_queue")
        reasons = {item["reason"] for item in result["violations"]}
        self.assertIn("missing_glossary_target", reasons)
        self.assertEqual(result["violations"][0]["expected"], RESOLVED)

    def test_a_casing_slip_is_repaired_rather_than_rejected(self):
        """Casing belongs to the resolver, so it is fixed, not treated as disagreement."""
        result = self._gate("Ve en modo de video.")
        self.assertTrue(result["repaired"])
        self.assertIn(RESOLVED, result["normalized"])
        self.assertFalse(result["blocked"])

    def test_a_conflicting_suggestion_is_never_rewritten_into_compliance(self):
        """Blocking hands the disagreement to a human; it must not silently substitute."""
        suggestion = "Ve en Modo película."
        result = self._gate(suggestion)
        self.assertEqual(result["normalized"], suggestion)
        self.assertNotIn(RESOLVED, result["normalized"])

    def test_title_row_keeps_the_term_unwrapped_and_passes(self):
        """Bracket policy is applied per row type, exactly as the translate path does."""
        result = self._gate(f"[{RESOLVED}]", row_key="title")
        self.assertEqual(result["normalized"], RESOLVED)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["action"], "propose")

    def test_an_empty_card_cannot_block_anything(self):
        result = self.checker.validate_audit_suggestion("Cualquier texto", {}, glossary_context={})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["status"], "pass")


if __name__ == "__main__":
    unittest.main()
