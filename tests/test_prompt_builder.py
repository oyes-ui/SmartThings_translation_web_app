# -*- coding: utf-8 -*-

import unittest

from fastapi.testclient import TestClient

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

