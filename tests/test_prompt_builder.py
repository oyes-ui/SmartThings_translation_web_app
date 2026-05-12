# -*- coding: utf-8 -*-

import unittest

from fastapi.testclient import TestClient

from checker_service import TranslationChecker
from main import app
from prompt_builder import PromptBuilder


class PromptBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder()

    def test_language_rule_selection(self):
        german = self.builder.describe_applied_modules(target_lang="German", source_lang="Korean")
        japanese = self.builder.describe_applied_modules(target_lang="Japanese", source_lang="Korean")
        english = self.builder.describe_applied_modules(target_lang="English", source_lang="Korean")

        self.assertTrue(german["language"]["active"])
        self.assertIn("Du/Sie", german["language"]["name"])
        self.assertTrue(japanese["language"]["active"])
        self.assertIn("Desu/Masu", japanese["language"]["name"])
        self.assertFalse(english["language"]["active"])

    def test_translation_prompt_modules(self):
        prompt = self.builder.build_translation_prompt(
            target_lang="Japanese",
            source_lang="Korean",
            bx_style_on=True,
            rag_context="Example 1: Source / Translation",
            row_key="hero_title",
            glossary_context={"SmartThings": "SmartThings"},
        )

        self.assertIn("COMMON LOCALIZATION STANDARD", prompt)
        self.assertIn("Japanese Politeness", prompt)
        self.assertIn("SAMSUNG BX STYLE", prompt)
        self.assertIn("Example 1", prompt)
        self.assertIn("without brackets", prompt)
        self.assertIn('"translation"', prompt)

    def test_japanese_description_context_uses_corner_brackets(self):
        formatting = self.builder.build_input_formatting("Japanese")
        self.assertEqual(formatting["glossary_prefix"], "「")
        self.assertEqual(formatting["glossary_suffix"], "」")
        self.assertTrue(self.builder.should_wrap_glossary("description_01"))

        prompt = self.builder.build_translation_prompt(
            target_lang="Japanese",
            source_lang="Korean",
            row_key="description_01",
            glossary_context={"SmartThings": "SmartThings"},
        )
        modules = self.builder.describe_applied_modules(
            target_lang="Japanese",
            source_lang="Korean",
            row_key="description_01",
            glossary_available=True,
        )

        self.assertIn("Japanese Bracket Style", prompt)
        self.assertIn("Use Japanese corner brackets 「」 instead of square brackets []", prompt)
        self.assertIn("brackets: 「」", modules["formatting"]["description"])
        self.assertIn("wrap_for_description", modules["formatting"]["description"])

    def test_title_context_skips_brackets_for_japanese(self):
        self.assertFalse(self.builder.should_wrap_glossary("hero_title"))

        prompt = self.builder.build_translation_prompt(
            target_lang="Japanese",
            source_lang="Korean",
            row_key="hero_title",
            glossary_context={"SmartThings": "SmartThings"},
        )
        modules = self.builder.describe_applied_modules(
            target_lang="Japanese",
            source_lang="Korean",
            row_key="hero_title",
            glossary_available=True,
        )

        self.assertIn("without brackets", prompt)
        self.assertIn("brackets: 「」", modules["formatting"]["description"])
        self.assertIn("skip_for_title_button", modules["formatting"]["description"])

    def test_japanese_locale_alias_uses_corner_bracket_rule(self):
        prompt = self.builder.build_translation_prompt(
            target_lang="일본",
            source_lang="Korean",
            row_key="description_01",
            glossary_context={"SmartThings": "SmartThings"},
        )

        self.assertIn("Japanese Bracket Style", prompt)
        self.assertIn("Wrap glossary terms in '「' and '」'", prompt)

    def test_audit_prompt_contract(self):
        prompt = self.builder.build_audit_prompt(
            source_lang="Korean",
            target_lang="German",
            target_lang_code="독어_독일",
            glossary_context={"AI": "KI"},
        )

        self.assertIn('"evaluation"', prompt)
        self.assertIn('"is_excellent"', prompt)
        self.assertIn('"suggested_fix"', prompt)
        self.assertIn("현지화 품질", prompt)

    def test_checker_glossary_brackets_respect_row_key_context(self):
        checker = TranslationChecker()
        checker.glossary = {
            "SmartThings": {
                "targets": {"일본": "SmartThings"},
                "rule": "",
            }
        }
        checker._compile_glossary_re()

        self.assertEqual(
            checker._check_glossary_brackets(
                "SmartThings를 실행하세요",
                "SmartThingsを起動",
                "일본",
                "Japanese",
                row_key="hero_title",
            ),
            [],
        )
        self.assertTrue(
            checker._check_glossary_brackets(
                "SmartThings를 실행하세요",
                "SmartThingsを起動できます",
                "일본",
                "Japanese",
                row_key="description_01",
            )
        )


class PromptModuleApiTests(unittest.TestCase):
    def test_prompt_modules_endpoint(self):
        client = TestClient(app)
        response = client.get(
            "/api/prompt_modules",
            params={
                "target_lang": "German",
                "source_lang": "Korean",
                "bx_style_on": "true",
                "glossary_available": "true",
                "row_key": "description_01",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["common"]["active"])
        self.assertTrue(payload["language"]["active"])
        self.assertTrue(payload["bx"]["active"])
        self.assertTrue(payload["glossary"]["active"])
        self.assertIn("German", payload["target_lang"])


if __name__ == "__main__":
    unittest.main()
