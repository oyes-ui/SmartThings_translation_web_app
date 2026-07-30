# -*- coding: utf-8 -*-
"""Proves the Markdown rule files reproduce the legacy constants exactly.

TEMPORARY. Delete this whole file in the same commit that removes the legacy
constants from prompt_modules.py (LANGUAGE_LOCALIZATION_RULES,
LANGUAGE_RULE_LABELS, BX_STYLE_RULES, COMMON_LOCALIZATION_STANDARD,
TYPOGRAPHY_AND_PUNCTUATION_RULES, GLOSSARY_TERM_RULES, the GLOSSARY_*
instruction strings, AUDIT_INTRO, AUDIT_CHECKLIST_RULES, AUDIT_GRADE_CRITERIA).
Its only job is to show the migration changed no prompt byte; once the legacy
constants are gone there is nothing left to compare against, and the durable
coverage lives in tests/test_rules_loader.py and tests/test_prompt_builder.py.

The grade-enum test in GradeEnumContractTests is the exception: it compares
decoders against each other, not against a legacy constant, so it must be moved
to test_rules_loader.py rather than deleted.
"""

import inspect
import unittest

from translation_web_app import prompt_builder as prompt_builder_module
from translation_web_app.prompt_builder import PromptBuilder
from translation_web_app.prompt_modules import (
    AUDIT_CHECKLIST_RULES,
    AUDIT_GRADE_CRITERIA,
    AUDIT_INTRO,
    BX_STYLE_RULES,
    COMMON_LOCALIZATION_STANDARD,
    GLOSSARY_BRACKET_WRAP_RULE,
    GLOSSARY_DISCLAIMER_NAV_EXCEPTION,
    GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE,
    GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_EAST_ASIAN,
    GLOSSARY_NO_BRACKET_INSTRUCTION,
    GLOSSARY_TERM_RULES,
    LANGUAGE_LOCALIZATION_RULES,
    LANGUAGE_RULE_LABELS,
    SHEET_CODE_LANGUAGE_ALIASES,
    TYPOGRAPHY_AND_PUNCTUATION_RULES,
)
from translation_web_app.rules_loader import load_rules


class LanguageRuleEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.bundle = load_rules()
        self.builder = PromptBuilder()

    def test_every_legacy_language_has_a_rules_file(self):
        # Subset, not equality: locales added after the migration (e.g.
        # Spanish_Colombia) have no legacy constant to compare against.
        self.assertLessEqual(set(LANGUAGE_LOCALIZATION_RULES), set(self.bundle.languages))

    def test_rule_text_is_identical_for_every_language(self):
        for key, legacy in LANGUAGE_LOCALIZATION_RULES.items():
            with self.subTest(language=key):
                self.assertEqual(list(self.bundle.languages[key].prompt_rules()), legacy)

    def test_display_names_are_identical(self):
        migrated = {
            key: name
            for key, name in self.bundle.display_names().items()
            if key in LANGUAGE_RULE_LABELS
        }
        self.assertEqual(migrated, LANGUAGE_RULE_LABELS)

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


class DocEquivalenceTests(unittest.TestCase):
    """common / typography / glossary / audit md == the legacy constants."""

    def setUp(self):
        self.bundle = load_rules()

    def test_common_standard(self):
        doc = self.bundle.doc("common")
        self.assertEqual(list(doc.texts("standard")), COMMON_LOCALIZATION_STANDARD["rules"])
        self.assertEqual(doc.display_name, COMMON_LOCALIZATION_STANDARD["name"])

    def test_typography(self):
        doc = self.bundle.doc("typography")
        self.assertEqual(list(doc.texts("rule")), TYPOGRAPHY_AND_PUNCTUATION_RULES["rules"])
        # display_name is emitted as the prompt section heading, so it is load-bearing.
        self.assertEqual(doc.display_name, TYPOGRAPHY_AND_PUNCTUATION_RULES["name"])

    def test_glossary_slots(self):
        doc = self.bundle.doc("glossary")
        self.assertEqual(doc.one("term_rule"), GLOSSARY_TERM_RULES["rules"][0])
        self.assertEqual(doc.one("bracket_wrap"), GLOSSARY_BRACKET_WRAP_RULE)
        self.assertEqual(doc.one("nav_exception"), GLOSSARY_DISCLAIMER_NAV_EXCEPTION)
        self.assertEqual(doc.one("no_bracket"), GLOSSARY_NO_BRACKET_INSTRUCTION)
        self.assertEqual(doc.one("nav_quote_default"), GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE)
        self.assertEqual(
            doc.one("nav_quote_east_asian"), GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_EAST_ASIAN
        )

    def test_audit_intro_checklist_and_grades(self):
        doc = self.bundle.doc("audit")
        self.assertEqual(doc.one("intro"), AUDIT_INTRO)
        self.assertEqual(list(doc.labelled("checklist")), list(AUDIT_CHECKLIST_RULES))
        self.assertEqual(list(doc.labelled("grade")), list(AUDIT_GRADE_CRITERIA.items()))


class SingleActiveSourceTests(unittest.TestCase):
    def test_prompt_builder_does_not_reference_legacy_constants(self):
        """Exactly one active source: the md files, not the kept-but-legacy dicts."""
        source = inspect.getsource(prompt_builder_module)
        for name in (
            "LANGUAGE_LOCALIZATION_RULES", "BX_STYLE_RULES", "LANGUAGE_RULE_LABELS",
            "COMMON_LOCALIZATION_STANDARD", "TYPOGRAPHY_AND_PUNCTUATION_RULES",
            "GLOSSARY_TERM_RULES", "GLOSSARY_BRACKET_WRAP_RULE", "GLOSSARY_NO_BRACKET_INSTRUCTION",
            "GLOSSARY_DISCLAIMER_NAV_", "AUDIT_INTRO", "AUDIT_CHECKLIST_RULES",
            "AUDIT_GRADE_CRITERIA",
        ):
            with self.subTest(constant=name):
                self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
