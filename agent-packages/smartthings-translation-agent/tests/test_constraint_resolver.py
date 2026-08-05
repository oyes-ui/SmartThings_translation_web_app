from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT.parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from translation_web_app.constraint_resolver import ConstraintResolver, OccurrenceActivationManifest  # noqa: E402


class ConstraintResolverTests(unittest.TestCase):
    def setUp(self):
        self.glossary = {
            "Air Care": {"targets": {"es_CO": "Air Care"}, "rule": ""},
            "Now brief": {"targets": {"es_CO": "Now Brief"}, "rule": ""},
            "Family Care": {"targets": {"es_CO": "Cuidado familiar"}, "rule": ""},
            "Care on call": {"targets": {"es_CO": "Care on call"}, "rule": "no bracket"},
            "Refrigerator": {"targets": {"es_CO": "Refrigerador"}, "rule": ""},
            "Service": {"targets": {"es_CO": "Servicios"}, "rule": ""},
        }
        self.resolver = ConstraintResolver(
            glossary=self.glossary,
            get_relevant_terms=lambda source: [term for term in self.glossary if term.lower() in source.lower()],
            get_target=lambda targets, code: targets.get(code),
            is_deactivated=lambda rule: "deactivated" in rule,
            has_exempt_marker=lambda rule: "no bracket" in rule,
            get_context_mode=lambda key: "title_button" if key.endswith(("title", "button")) else "disclaimer" if "disclaimer" in key else "body",
        )

    def test_title_keeps_lexical_casing_but_removes_default_wrap(self):
        item = self.resolver.resolve("Now brief", "es_CO", row_key="title")[0]
        self.assertEqual((item.target, item.bracket_policy), ("Now Brief", "no_bracket"))
        self.assertEqual(item.no_bracket_reasons, ("section_role",))

    def test_navigation_path_and_glossary_exempt_are_no_bracket(self):
        nav = self.resolver.resolve("Family Care", "es_CO", row_key="disclaimer", inside_navigation_path=True)[0]
        exempt = self.resolver.resolve("Care on call", "es_CO", row_key="body")[0]
        self.assertIn("navigation_path", nav.no_bracket_reasons)
        self.assertIn("glossary_exempt", exempt.no_bracket_reasons)

    def test_description_wrap_and_validator_blocks_wrong_lexical_form(self):
        item = self.resolver.resolve("Refrigerator Service", "es_CO", row_key="description")
        self.assertEqual([x.bracket_policy for x in item], ["wrap", "wrap"])
        verdict = self.resolver.validate_target("[refrigerador] [Servicio]", item)
        self.assertEqual(verdict["status"], "blocked")

    def test_occurrence_manifest_overrides_legacy_deactivation_only(self):
        self.glossary["Air Care"]["rule"] = "deactivated"
        manifest = OccurrenceActivationManifest([{"story": "047", "cell": "C7", "source_term": "Air Care", "active": True}])
        resolver = ConstraintResolver(**{**self.resolver.__dict__, "activation_manifest": manifest})
        item = resolver.resolve("Air Care", "es_CO", story="047", cell="C7")[0]
        self.assertTrue(item.active)
        self.assertEqual(item.target, "Air Care")


if __name__ == "__main__":
    unittest.main()
