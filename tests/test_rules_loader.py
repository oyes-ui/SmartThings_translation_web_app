# -*- coding: utf-8 -*-
"""Schema and validation tests for the Markdown rule loader."""

import tempfile
import textwrap
import unittest
from pathlib import Path

from translation_web_app.prompt_modules import SHEET_CODE_LANGUAGE_ALIASES
from translation_web_app.rules_loader import RuleFileError, load_rules


VALID_LANGUAGE = """\
---
schema_version: 1
kind: language
canonical_key: German
display_name: German Du-form Consistency
rule_order: sequence
rules:
- rule_id: german-001
  scope: [app_prompt]
  text: Use Du-form consistently.
- rule_id: german-002
  scope: [agent_audit]
  severity: major
  text: Check Du-form conjugation consistency.
---

# German
"""

VALID_BX = """\
---
schema_version: 1
kind: bx_style
canonical_key: bx_style
display_name: Samsung BX Style
rule_order: sequence
identity:
  role: Samsung BX Writer & Translator
  persona: Confident Explorer
  goal: Craft confident copy.
rules:
- rule_id: bx-open-001
  group: OPEN
  scope: [app_prompt]
  text: Go beyond the literal benefit.
- rule_id: bx-negative-001
  group: NEGATIVE
  scope: [app_prompt]
  text: Do NOT use negative framing.
examples:
- type: OPEN (Headlines)
  input: Turn on the lights.
  output: Lights? On.
---

# BX
"""


class RuleLoaderTestCase(unittest.TestCase):
    """Builds a throwaway rules tree so no tracked file is ever broken."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "languages").mkdir()
        self.write_language(VALID_LANGUAGE)
        self.write_bx(VALID_BX)
        self.addCleanup(self._tmp.cleanup)

    def write_language(self, text, name="German.md"):
        (self.root / "languages" / name).write_text(text, encoding="utf-8")

    def write_bx(self, text):
        (self.root / "bx_style.md").write_text(text, encoding="utf-8")

    def assertRuleError(self, fragment):
        with self.assertRaises(RuleFileError) as ctx:
            load_rules(self.root)
        self.assertIn(fragment, str(ctx.exception))


class LoadValidRulesTests(RuleLoaderTestCase):
    def test_loads_valid_tree(self):
        bundle = load_rules(self.root)
        self.assertEqual(set(bundle.languages), {"German"})
        self.assertEqual(bundle.languages["German"].display_name, "German Du-form Consistency")

    def test_scope_filtering_keeps_agent_rules_out_of_prompts(self):
        bundle = load_rules(self.root)
        self.assertEqual(bundle.languages["German"].prompt_rules(), ("Use Du-form consistently.",))
        audit = bundle.languages["German"].scoped("agent_audit")
        self.assertEqual([rule.rule_id for rule in audit], ["german-002"])

    def test_rules_for_scope_reaches_agent_audit_rules(self):
        found = load_rules(self.root).rules_for_scope("agent_audit")
        self.assertEqual([(key, rule.rule_id) for key, rule in found], [("German", "german-002")])

    def test_body_is_captured_but_not_parsed(self):
        self.assertIn("# German", load_rules(self.root).languages["German"].body)

    def test_draft_rules_are_excluded_from_prompts(self):
        self.write_language(VALID_LANGUAGE.replace(
            "  text: Use Du-form consistently.",
            "  status: draft\n  text: Use Du-form consistently.",
        ))
        self.assertEqual(load_rules(self.root).languages["German"].prompt_rules(), ())

    def test_crlf_and_bom_are_normalized(self):
        self.write_language("﻿" + VALID_LANGUAGE.replace("\n", "\r\n"))
        bundle = load_rules(self.root)
        self.assertEqual(bundle.languages["German"].prompt_rules(), ("Use Du-form consistently.",))


class LanguageValidationTests(RuleLoaderTestCase):
    def test_missing_front_matter(self):
        self.write_language("# German\n\nNo front matter here.\n")
        self.assertRuleError("missing YAML front matter")

    def test_unsupported_schema_version(self):
        self.write_language(VALID_LANGUAGE.replace("schema_version: 1", "schema_version: 2"))
        self.assertRuleError("unsupported schema_version 2")

    def test_missing_display_name(self):
        self.write_language(VALID_LANGUAGE.replace(
            "display_name: German Du-form Consistency\n", ""))
        self.assertRuleError("missing required field 'display_name'")

    def test_canonical_key_must_match_filename(self):
        self.write_language(VALID_LANGUAGE.replace("canonical_key: German", "canonical_key: Deutsch"))
        self.assertRuleError("does not match filename stem 'German'")

    def test_duplicate_yaml_key(self):
        self.write_language(VALID_LANGUAGE.replace(
            "display_name: German Du-form Consistency",
            "display_name: First\ndisplay_name: Second",
        ))
        self.assertRuleError("duplicate key 'display_name'")

    def test_unknown_front_matter_field(self):
        self.write_language(VALID_LANGUAGE.replace(
            "rule_order: sequence", "rule_order: sequence\ndisplayname: typo"))
        self.assertRuleError("unknown front matter field 'displayname'")

    def test_unknown_rule_field(self):
        self.write_language(VALID_LANGUAGE.replace(
            "  scope: [app_prompt]\n  text: Use Du-form consistently.",
            "  scope: [app_prompt]\n  sevrity: major\n  text: Use Du-form consistently.",
        ))
        self.assertRuleError("has unknown field 'sevrity'")

    def test_unsupported_scope(self):
        self.write_language(VALID_LANGUAGE.replace("scope: [app_prompt]", "scope: [app-prompt]", 1))
        self.assertRuleError("'app-prompt' is not a supported scope")

    def test_unsupported_rule_order(self):
        self.write_language(VALID_LANGUAGE.replace("rule_order: sequence", "rule_order: severity"))
        self.assertRuleError("rule_order 'severity' is not supported")

    def test_invalid_yaml_reports_line(self):
        self.write_language(VALID_LANGUAGE.replace(
            "  text: Use Du-form consistently.",
            "  text: Unbalanced: quoting: here",
        ))
        self.assertRuleError("invalid YAML front matter at line")

    def test_duplicate_rule_id_across_files(self):
        self.write_language(VALID_LANGUAGE.replace("canonical_key: German", "canonical_key: Polish"),
                            name="Polish.md")
        self.assertRuleError("rule_id 'german-001' is already defined by")

    def test_empty_language_directory(self):
        (self.root / "languages" / "German.md").unlink()
        self.assertRuleError("no rule files found")

    def test_errors_are_aggregated(self):
        self.write_language(VALID_LANGUAGE.replace("schema_version: 1", "schema_version: 9"))
        self.write_bx(VALID_BX.replace("schema_version: 1", "schema_version: 9"))
        with self.assertRaises(RuleFileError) as ctx:
            load_rules(self.root)
        self.assertIn("2 problem(s)", str(ctx.exception))


class BxValidationTests(RuleLoaderTestCase):
    def test_missing_bx_file(self):
        (self.root / "bx_style.md").unlink()
        self.assertRuleError("bx_style.md: file not found")

    def test_incomplete_identity(self):
        self.write_bx(VALID_BX.replace("  goal: Craft confident copy.\n", ""))
        self.assertRuleError("identity is missing required key 'goal'")

    def test_unsupported_group(self):
        self.write_bx(VALID_BX.replace("group: OPEN", "group: CALM"))
        self.assertRuleError("group 'CALM' is not supported")

    def test_malformed_example(self):
        self.write_bx(VALID_BX.replace("  output: Lights? On.\n", ""))
        self.assertRuleError("must be a mapping with keys type, input, output")

    def test_voice_attribute_order_follows_file_order(self):
        bundle = load_rules(self.root)
        self.assertEqual(list(bundle.bx.voice_attributes), ["OPEN"])
        self.assertEqual(bundle.bx.negative_constraints, ("Do NOT use negative framing.",))


class RealRuleFilesTests(unittest.TestCase):
    """Invariants over the committed rule files, not a temp fixture."""

    def test_every_sheet_alias_resolves_to_a_rules_file(self):
        bundle = load_rules()
        for code, canonical_key in SHEET_CODE_LANGUAGE_ALIASES.items():
            with self.subTest(sheet_code=code):
                self.assertIn(canonical_key, bundle.languages)

    def test_every_rule_id_is_unique(self):
        bundle = load_rules()
        rule_ids = [rule.rule_id for lang in bundle.languages.values() for rule in lang.rules]
        rule_ids += [rule.rule_id for rule in bundle.bx.rules]
        self.assertEqual(len(rule_ids), len(set(rule_ids)))

    def test_every_language_has_at_least_one_prompt_rule(self):
        for key, language in load_rules().languages.items():
            with self.subTest(language=key):
                self.assertTrue(language.prompt_rules())


if __name__ == "__main__":
    unittest.main()
