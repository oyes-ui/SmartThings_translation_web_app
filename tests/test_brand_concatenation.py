# -*- coding: utf-8 -*-
"""
Brand-concatenation lint tests (dropped-space detection).

Covers the reported failure mode (``SmartThings Family Care`` ->
``SmartThingsFamily Care``, ``Galaxy Device control`` -> ``GalaxyDevice control``)
and, critically, the false-positive exclusions the briefing flagged: Turkish
agglutination, Russian case endings, and CJK / Korean particle attachment must
NOT trigger, without any per-language allow-list.
"""
import unittest

from translation_web_app.checker_service import TranslationChecker


def _checker(glossary):
    ck = TranslationChecker()
    ck.glossary = glossary
    ck._compile_glossary_re()
    return ck


SMARTTHINGS = {"SmartThings": {"targets": {"ru_RU": "SmartThings", "tr_TR": "SmartThings",
                                           "ja_JP": "SmartThings", "pt_BR": "SmartThings",
                                           "de_DE": "SmartThings"}, "rule": ""}}
GALAXY = {"Galaxy Device": {"targets": {"pt_BR": "Galaxy Device"}, "rule": ""}}


class BrandConcatenationPositiveTests(unittest.TestCase):
    def test_flags_missing_space_after_brand(self):
        ck = _checker(SMARTTHINGS)
        issues = ck._check_brand_concatenation(
            "SmartThings Family Care", "SmartThingsFamily Care", "pt_BR", "Portuguese_Latin"
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("SmartThingsFamily", issues[0])

    def test_flags_multiword_brand_token_glue(self):
        ck = _checker(GALAXY)
        issues = ck._check_brand_concatenation(
            "Galaxy Device control", "GalaxyDevice control", "pt_BR", "Portuguese_Latin"
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("GalaxyDevice", issues[0])

    def test_flags_preceding_word_glued_to_brand(self):
        ck = _checker(SMARTTHINGS)
        issues = ck._check_brand_concatenation(
            "open SmartThings now", "openSmartThings now", "de_DE", "German"
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("openSmartThings", issues[0])

    def test_reports_each_merged_fragment_once(self):
        ck = _checker({**SMARTTHINGS, "Family Care": {"targets": {"pt_BR": "Family Care"}, "rule": ""}})
        issues = ck._check_brand_concatenation(
            "SmartThings Family Care", "SmartThingsFamily Care", "pt_BR", "Portuguese_Latin"
        )
        # "SmartThings" (after-glue) and "Family" (before-glue) describe the SAME fragment.
        self.assertEqual(len(issues), 1)


class BrandConcatenationNoFalsePositiveTests(unittest.TestCase):
    def test_correct_spacing_is_clean(self):
        ck = _checker(SMARTTHINGS)
        self.assertEqual(
            ck._check_brand_concatenation("SmartThings Family Care", "SmartThings Family Care",
                                          "pt_BR", "Portuguese_Latin"),
            [],
        )

    def test_turkish_apostrophe_suffix_not_flagged(self):
        ck = _checker(SMARTTHINGS)
        self.assertEqual(
            ck._check_brand_concatenation("Set it in SmartThings.", "SmartThings'te ayarlayın.",
                                          "tr_TR", "Turkish"),
            [],
        )

    def test_turkish_lowercase_suffix_not_flagged(self):
        ck = _checker(SMARTTHINGS)
        self.assertEqual(
            ck._check_brand_concatenation("SmartThings app", "SmartThingste", "tr_TR", "Turkish"),
            [],
        )

    def test_russian_cyrillic_case_ending_not_flagged(self):
        ck = _checker(SMARTTHINGS)
        # Cyrillic ending glued to the brand is normal declension, not a dropped space.
        self.assertEqual(
            ck._check_brand_concatenation("Works with SmartThings.", "работает с SmartThingsом.",
                                          "ru_RU", "Russian"),
            [],
        )

    def test_cjk_attachment_not_flagged(self):
        ck = _checker(SMARTTHINGS)
        self.assertEqual(
            ck._check_brand_concatenation("Set in SmartThings.", "SmartThings機能を設定します。",
                                          "ja_JP", "Japanese"),
            [],
        )

    def test_korean_particle_not_flagged(self):
        ck = _checker(SMARTTHINGS)
        self.assertEqual(
            ck._check_brand_concatenation("Launch SmartThings", "SmartThings를 실행하세요",
                                          "ko_KR", "Korean"),
            [],
        )

    def test_common_noun_lowercased_target_not_flagged(self):
        # A capitalized glossary target used as a lowercase common noun in prose must not
        # match (case-sensitive), avoiding the Doorbell->campainha style false positives.
        ck = _checker({"Doorbell": {"targets": {"pt_BR": "Campainha"}, "rule": ""}})
        self.assertEqual(
            ck._check_brand_concatenation("Doorbell rings.", "A descampainha toca na TV.",
                                          "pt_BR", "Portuguese_Latin"),
            [],
        )

    def test_deactivated_term_not_considered(self):
        ck = _checker({"SmartThings": {"targets": {"pt_BR": "SmartThings"}, "rule": "비활성화"}})
        self.assertEqual(
            ck._check_brand_concatenation("SmartThings Family", "SmartThingsFamily", "pt_BR",
                                          "Portuguese_Latin"),
            [],
        )

    def test_no_glossary_no_issues(self):
        ck = _checker({})
        self.assertEqual(ck._check_brand_concatenation("x", "SmartThingsFamily", "pt_BR", "x"), [])

    def test_camelcase_brand_prefix_not_flagged(self):
        # Regression: a short brand token ("Smart", from "Smart TV") must not flag the
        # legitimate single-token brand "SmartThings" it prefixes. Observed as 8 false
        # positives on shipped RU/DE/FR/... rows before the glossary-word whitelist.
        ck = _checker({
            "Smart TV": {"targets": {"de_DE": "Smart TV"}, "rule": ""},
            "SmartThings": {"targets": {"de_DE": "SmartThings"}, "rule": ""},
        })
        self.assertEqual(
            ck._check_brand_concatenation(
                "Samsung Smart TV supports SmartThings.",
                "Samsung Smart TV unterstützt SmartThings.",
                "de_DE", "German",
            ),
            [],
        )

    def test_prefix_whitelist_still_flags_real_drop(self):
        # The whitelist must not mask a genuine dropped space: "SmartThingsFamily" is not a
        # glossary word, so it is still flagged even when "Smart"/"SmartThings" are terms.
        ck = _checker({
            "Smart TV": {"targets": {"pt_BR": "Smart TV"}, "rule": ""},
            "SmartThings": {"targets": {"pt_BR": "SmartThings"}, "rule": ""},
        })
        issues = ck._check_brand_concatenation(
            "SmartThings Family Care", "SmartThingsFamily Care", "pt_BR", "Portuguese_Latin"
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("SmartThingsFamily", issues[0])


if __name__ == "__main__":
    unittest.main()
