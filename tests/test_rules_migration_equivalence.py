# -*- coding: utf-8 -*-
"""Proves the Markdown rule files reproduce the legacy constants exactly.

TEMPORARY. Delete this whole file in the same commit that removes
LANGUAGE_LOCALIZATION_RULES, LANGUAGE_RULE_LABELS and BX_STYLE_RULES from
prompt_modules.py. Its only job is to show the migration changed no prompt
byte; once the legacy constants are gone there is nothing left to compare
against, and the durable coverage lives in tests/test_rules_loader.py and
tests/test_prompt_builder.py.
"""

import inspect
import unittest

from translation_web_app import prompt_builder as prompt_builder_module
from translation_web_app.prompt_builder import PromptBuilder
from translation_web_app.prompt_modules import (
    BX_STYLE_RULES,
    LANGUAGE_LOCALIZATION_RULES,
    LANGUAGE_RULE_LABELS,
    SHEET_CODE_LANGUAGE_ALIASES,
)
from translation_web_app.rules_loader import load_rules


class LanguageRuleEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.bundle = load_rules()
        self.builder = PromptBuilder()

    def test_every_legacy_language_has_a_rules_file(self):
        self.assertEqual(set(self.bundle.languages), set(LANGUAGE_LOCALIZATION_RULES))

    def test_rule_text_is_identical_for_every_language(self):
        for key, legacy in LANGUAGE_LOCALIZATION_RULES.items():
            with self.subTest(language=key):
                self.assertEqual(list(self.bundle.languages[key].prompt_rules()), legacy)

    def test_display_names_are_identical(self):
        self.assertEqual(self.bundle.display_names(), LANGUAGE_RULE_LABELS)

    def test_rendered_language_section_is_identical(self):
        """The real guarantee: same bytes reach the model, both headings."""
        for key, legacy in LANGUAGE_LOCALIZATION_RULES.items():
            for korean in (False, True):
                heading = "[언어별 현지화 기준]" if korean else "[LANGUAGE SPECIFIC RULE]"
                expected = f"{heading}\n{LANGUAGE_RULE_LABELS[key]}\n" + "\n".join(
                    f"- {rule}" for rule in legacy
                )
                with self.subTest(language=key, korean_heading=korean):
                    self.assertEqual(
                        self.builder._build_language_section(key, korean_heading=korean),
                        expected,
                    )

    def test_sheet_codes_render_identically(self):
        """Covers the alias + fuzzy-match path, not just exact keys."""
        for code, key in SHEET_CODE_LANGUAGE_ALIASES.items():
            with self.subTest(sheet_code=code):
                self.assertEqual(
                    self.builder._build_language_section(f"{code}(market)"),
                    self.builder._build_language_section(key),
                )


class BxStyleEquivalenceTests(unittest.TestCase):
    def test_bx_reconstructs_the_legacy_structure(self):
        self.assertEqual(load_rules().bx.as_legacy_dict(), BX_STYLE_RULES)

    def test_voice_attribute_order_is_preserved(self):
        # dict equality ignores order, but _build_bx_section iterates this mapping.
        self.assertEqual(
            list(load_rules().bx.voice_attributes),
            list(BX_STYLE_RULES["voice_attributes"]),
        )


class SingleActiveSourceTests(unittest.TestCase):
    def test_prompt_builder_does_not_reference_legacy_constants(self):
        """Exactly one active source: the md files, not the kept-but-legacy dicts."""
        source = inspect.getsource(prompt_builder_module)
        self.assertNotIn("LANGUAGE_LOCALIZATION_RULES", source)
        self.assertNotIn("BX_STYLE_RULES", source)
        self.assertNotIn("LANGUAGE_RULE_LABELS", source)


if __name__ == "__main__":
    unittest.main()
