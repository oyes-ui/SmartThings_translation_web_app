# -*- coding: utf-8 -*-
"""
Glossary bracket-policy tests (story 053 / C17·C12).

Covers the single bracket-policy resolver, navigation-path span detection across
quote styles, the wrongly-present bracket audit, and the span-scoped strip.

The false-positive regression exercises REAL disclaimer source/target rows from the
project RAG DB (runtime/rag_db/rag_store.db) — the deterministic audit must not flag
correct shipped translations. It skips automatically when the DB/glossary are absent.
"""

import asyncio
import os
import sqlite3
import unittest

from translation_web_app.checker_service import TranslationChecker
from translation_web_app.prompt_builder import PromptBuilder

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOSSARY_CSV = os.path.join(_ROOT, "runtime", "glossary", "latest_glossary.csv")
RAG_DB = os.path.join(_ROOT, "runtime", "rag_db", "rag_store.db")

# RAG sheet-name -> (glossary target code, prompt language name)
_LANG = {
    "RU(러시아)": ("ru_RU", "Russian"),
    "BR(브라질)": ("pt_BR", "Portuguese_Latin"),
    "CN(중국)": ("zh_CN", "Simplified Chinese"),
    "DE(독일)": ("de_DE", "German"),
    "JA(일본)": ("ja_JP", "Japanese"),
    "FR(프랑스)": ("fr_FR", "French"),
    "IT(이탈리아)": ("it_IT", "Italian"),
    "ES(스페인)": ("es_ES", "Spanish"),
    "PL(폴란드)": ("pl_PL", "Polish"),
    "TR(터키)": ("tr_TR", "Turkish"),
}

DISCLAIMER = "//section_053_2_disclaimer"


class BracketPolicyResolverTests(unittest.TestCase):
    def setUp(self):
        self.pb = PromptBuilder()

    def test_policy_truth_table(self):
        cases = [
            # (row_key, rule, inside_nav_path) -> expected policy
            ((DISCLAIMER, "", True), "no_bracket"),          # nav path exception
            ((DISCLAIMER, "", False), "wrap"),               # disclaimer generic
            ((DISCLAIMER, "대괄호 제외", False), "no_bracket"),  # term exemption
            ((DISCLAIMER, "비활성화", False), "skip"),          # deactivated
            (("//section_053_7", "", False), "no_bracket"),  # digit-ending row_key == title
            (("//section_053_1_button", "", False), "no_bracket"),
            (("//section_053_5_description", "", False), "wrap"),
        ]
        for (row_key, rule, nav), expected in cases:
            self.assertEqual(
                self.pb.resolve_glossary_bracket_policy(row_key=row_key, rule_text=rule, inside_nav_path=nav),
                expected,
                msg=f"{row_key!r} rule={rule!r} nav={nav}",
            )

    def test_deactivation_and_exempt_helpers(self):
        self.assertTrue(self.pb.is_glossary_deactivated("비활성화"))
        self.assertTrue(self.pb.is_glossary_deactivated("Deactivate"))
        self.assertFalse(self.pb.is_glossary_deactivated("대괄호 제외"))
        self.assertTrue(self.pb.has_exempt_marker("대괄호 제외"))
        self.assertTrue(self.pb.has_exempt_marker("no bracket"))
        self.assertFalse(self.pb.has_exempt_marker(""))

    def test_should_wrap_glossary_delegates_to_resolver(self):
        self.assertTrue(self.pb.should_wrap_glossary("//section_053_5_description", ""))
        self.assertFalse(self.pb.should_wrap_glossary("//section_053_5_description", "대괄호 제외"))
        self.assertFalse(self.pb.should_wrap_glossary("//section_053_7", ""))


class NavigationPathSpanTests(unittest.TestCase):
    def setUp(self):
        self.checker = TranslationChecker()

    def test_detects_all_quote_styles(self):
        # Real quote styles observed in shipped RU/BR/JA disclaimer rows.
        samples = {
            "ascii": 'посмотреть в "SmartThings > Меню > Журнал ремонта".',
            "guillemets": 'включите «Жизнь > Family Care > Уход по вызову».',
            "fullwidth": 'ativado em “Aparelho > Ar-condicionado > Cuidado do ar”.',
            "japanese": 'テレビの「SmartThings Settings > 通知 > ドアチャイム」画面。',
        }
        for name, text in samples.items():
            self.assertTrue(self.checker._get_navigation_path_spans(text), msg=name)

    def test_ignores_quoted_text_without_breadcrumb(self):
        self.assertEqual(self.checker._get_navigation_path_spans('он сказал "привет".'), [])


class BracketAuditTests(unittest.TestCase):
    """Wrongly-present bracket detection (C17 / C12). Missing brackets are intentionally
    not flagged deterministically, so those cases must return []."""

    def _checker(self, source_term, target_code, target_value, rule=""):
        checker = TranslationChecker()
        checker.glossary = {source_term: {"targets": {target_code: target_value}, "rule": rule}}
        checker._compile_glossary_re()
        return checker

    def test_c17_bracket_inside_nav_path_is_flagged(self):
        ck = self._checker("Now brief", "ru_RU", "Мой день")
        bad = 'Откройте "Настройки > [Мой день]".'
        good = 'Откройте "Настройки > Мой день".'
        self.assertTrue(ck._check_glossary_brackets("Open Settings > Now brief.", bad, "ru_RU", "Russian", row_key=DISCLAIMER))
        self.assertEqual(ck._check_glossary_brackets("Open Settings > Now brief.", good, "ru_RU", "Russian", row_key=DISCLAIMER), [])

    def test_c12_exempt_term_bracketed_is_flagged(self):
        ck = self._checker("SmartThings", "pt_BR", "SmartThings", rule="대괄호 제외")
        self.assertTrue(ck._check_glossary_brackets("Manage SmartThings.", "Gerencie o [SmartThings].", "pt_BR", "Portuguese_Latin", row_key="//section_053_5_description"))

    def test_mixed_same_term_flags_only_inside_path(self):
        ck = self._checker("Now brief", "ru_RU", "Мой день")
        # Correctly bracketed outside the path + wrongly bracketed inside it.
        mixed = 'Режим [Мой день]. Откройте "Настройки > [Мой день]".'
        issues = ck._check_glossary_brackets('[Now brief]. Open "Settings > Now brief".', mixed, "ru_RU", "Russian", row_key=DISCLAIMER)
        self.assertEqual(len(issues), 1)
        self.assertIn("내비게이션 경로", issues[0])

    def test_missing_bracket_is_not_flagged(self):
        # Common-noun collisions (Doorbell->Campainha) must not raise false positives.
        ck = self._checker("Doorbell", "pt_BR", "Campainha")
        self.assertEqual(
            ck._check_glossary_brackets("Doorbell notifications.", "Notificações da campainha na TV.", "pt_BR", "Portuguese_Latin", row_key="//section_053_5_description"),
            [],
        )

    def test_c20_title_term_wrapped_in_quotes_is_flagged(self):
        # A standalone glossary term wrapped in quotation marks/guillemets in a title
        # is a violation (Russian «...» rule must not beat the title no-wrapper rule).
        ck = self._checker("Now brief", "ru_RU", "Мой день")
        for wrapped in ("«Мой день»", '"Мой день"', "“Мой день”"):
            self.assertTrue(
                ck._check_glossary_brackets("Now brief", wrapped, "ru_RU", "Russian", row_key="//section_053_1"),
                msg=wrapped,
            )

    def test_quotes_are_not_flagged_outside_title_context(self):
        # In a disclaimer, «...» legitimately quotes the navigation path -> no issue.
        ck = self._checker("Now brief", "ru_RU", "Мой день")
        self.assertEqual(
            ck._check_glossary_brackets("Open Settings > Now brief.", 'Откройте «Настройки > Мой день».', "ru_RU", "Russian", row_key=DISCLAIMER),
            [],
        )

    def test_title_quoted_path_not_flagged(self):
        # Quotes wrapping a whole path (term not standalone) are allowed even in a title.
        ck = self._checker("Now brief", "ru_RU", "Мой день")
        self.assertEqual(
            ck._check_glossary_brackets("Open Settings > Now brief", 'Открыть «Настройки > Мой день»', "ru_RU", "Russian", row_key="//section_053_1"),
            [],
        )


class BracketStripTests(unittest.TestCase):
    def setUp(self):
        self.checker = TranslationChecker()

    def test_title_button_strips_all(self):
        out = self.checker._strip_glossary_brackets_by_policy("[自宅分析] can be set in [Settings]", {"Home insight": "自宅分析", "Settings": "Settings"}, row_key="//section_045_1")
        self.assertEqual(out, "自宅分析 can be set in Settings")

    def test_disclaimer_strips_only_inside_nav_path(self):
        mixed = 'Режим [Мой день]. Откройте "Настройки > [Мой день]".'
        out = self.checker._strip_glossary_brackets_by_policy(mixed, {"Now brief": "Мой день"}, row_key=DISCLAIMER)
        self.assertEqual(out, 'Режим [Мой день]. Откройте "Настройки > Мой день".')

    def test_exempt_term_stripped_whole_cell(self):
        # 대괄호 제외 term wrongly bracketed anywhere is unwrapped (C12 generation fix).
        ctx = {"SmartThings": "SmartThings (EXCEPTION: Do NOT wrap 'SmartThings' in brackets)"}
        out = self.checker._strip_glossary_brackets_by_policy("Abra o [SmartThings] agora.", ctx, row_key="//section_053_5_description")
        self.assertEqual(out, "Abra o SmartThings agora.")

    def test_description_generic_term_preserved(self):
        out = self.checker._strip_glossary_brackets_by_policy("Use [Home insight] daily.", {"Home insight": "Home insight"}, row_key="//section_053_5_description")
        self.assertEqual(out, "Use [Home insight] daily.")

    def test_title_strips_quotation_wrappers(self):
        # C20: standalone glossary term wrapped in quotes/guillemets in a title is unwrapped.
        ctx = {"Now brief": "Мой день"}
        for wrapped in ("«Мой день»", '"Мой день"', "“Мой день”"):
            self.assertEqual(
                self.checker._strip_glossary_brackets_by_policy(wrapped, ctx, row_key="//section_053_1"),
                "Мой день",
                msg=wrapped,
            )

    def test_disclaimer_preserves_path_quotation_marks(self):
        # The path's own guillemets must survive the strip (only inner brackets are removed).
        ctx = {"Now brief": "Мой день"}
        text = 'Откройте «Настройки > [Мой день]».'
        self.assertEqual(
            self.checker._strip_glossary_brackets_by_policy(text, ctx, row_key=DISCLAIMER),
            'Откройте «Настройки > Мой день».',
        )


@unittest.skipUnless(os.path.exists(RAG_DB) and os.path.exists(GLOSSARY_CSV), "RAG DB / glossary not available")
class RealDataFalsePositiveRegression(unittest.TestCase):
    """The deterministic audit must produce ZERO issues on correct shipped translations."""

    def test_no_false_positives_on_real_disclaimer_targets(self):
        checker = TranslationChecker()
        asyncio.run(checker.load_glossary_from_file(GLOSSARY_CSV, "Korean"))

        placeholders = ",".join("?" * len(_LANG))
        conn = sqlite3.connect(RAG_DB)
        rows = conn.execute(
            "SELECT story_id, section_code, source_text, target_lang, target_text FROM rag_pairs "
            "WHERE section_code LIKE '%disclaimer%' AND source_text LIKE '%>%' "
            "AND target_lang IN (" + placeholders + ")",
            tuple(_LANG),
        ).fetchall()
        conn.close()
        self.assertGreater(len(rows), 0, "expected real disclaimer nav-path rows in RAG DB")

        false_positives = []
        for story, section, source, tlang, target in rows:
            code, name = _LANG[tlang]
            issues = checker._check_glossary_brackets(source, target, code, name, row_key=section)
            if issues:
                false_positives.append((story, section, tlang, issues))

        self.assertEqual(false_positives, [], msg=f"{len(false_positives)} false positives on shipped data")


if __name__ == "__main__":
    unittest.main()
