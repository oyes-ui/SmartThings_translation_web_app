"""The agent report and the app report must be the same document.

The agent package had grown a second renderer, so the same workbook produced two
differently shaped reports depending on which path reviewed it — defeating the
reason the agent path exists, which is to deliver the app's report *plus* the
sheet-level consistency the app cannot do.

These tests pin the seam: identical per-cell input renders identically through the
app, and everything the agent knows on top of that lands in the section the app's
renderer deliberately leaves empty.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

import agent_app_report  # noqa: E402
from review_report_builder import _GRADE_FOR_STATUS  # noqa: E402
from translation_web_app.report_builder import (  # noqa: E402
    render_finding, render_report, status_for_grade,
)

EVIDENCE = {
    "cell": "C18", "row_type": "button",
    "source_text": "Movie mode", "target_text": "Modo de video",
    "case_section": "별도 지적 사항 없음.",
    "glossary_section": "용어집 사전 감지:\n- Movie mode → Modo de video",
    "constraint_card": {"terms": [{"source_term": "Movie mode", "target": "Modo de video"}]},
    "constraint_validation": {"status": "pass"},
}


class CellSectionParityTests(unittest.TestCase):
    def _result(self, **overrides):
        return agent_app_report.build_cell_result(
            EVIDENCE, sheet="CO(콜롬비아)", grade="Needs Revision",
            evaluation=[{"category": "용어집", "comment": "확정 용어 유지"}],
            suggested_fix="Modo de video 2", **overrides)

    def test_cell_sections_are_byte_identical_to_a_direct_app_render(self):
        """The agent path adds front matter and notes; the audit itself is the app's."""
        result = self._result()
        through_agent = agent_app_report.render_through_app(
            None, title="검수", report_id="r", source_file_id="s", results=[result])
        directly = render_report(title="검수", front_matter={}, findings=[render_finding(result)])
        marker = "## 셀 검수"
        self.assertEqual(through_agent.split(marker, 1)[1].split("## Agent Notes")[0],
                         directly.split(marker, 1)[1].split("## Agent Notes")[0])

    def test_a_surviving_proposal_is_reported_as_applyable(self):
        """The report must not say not_applicable for a change the manifest ships."""
        self.assertIn("apply_status: pending_approval", render_finding(self._result()))

    def test_packet_sections_are_passed_through_not_recomposed(self):
        """The app composed these strings; rewording them here is what causes drift."""
        result = agent_app_report.build_cell_result(EVIDENCE, sheet="CO(콜롬비아)")
        self.assertEqual(result["glossary_section"], EVIDENCE["glossary_section"])
        self.assertEqual(result["case_section"], EVIDENCE["case_section"])

    def test_a_missing_back_translation_says_so_rather_than_reading_as_clean(self):
        result = agent_app_report.build_cell_result(EVIDENCE, sheet="CO(콜롬비아)")
        self.assertIn("역번역 미수행", result["back_translation"])


class GradeMappingTests(unittest.TestCase):
    """Every stage status must land on a grade the app actually recognises.

    status_for_grade reports an unknown grade as `blocked` — "the audit produced no
    verdict" — so sending `Pass`, which is not one of the app's labels, rendered a
    clean cell as blocked. Asserting only on Needs Revision hid that.
    """

    EXPECTED = {
        "pass": "pass", "warning": "warning", "needs_revision": "needs_revision",
        "blocked": "needs_revision", "glossary_activation_review": "needs_revision",
    }

    def test_each_stage_status_renders_as_its_own_status(self):
        for stage_status, rendered in self.EXPECTED.items():
            with self.subTest(stage_status):
                self.assertEqual(status_for_grade(_GRADE_FOR_STATUS[stage_status]), rendered)

    def test_no_mapping_falls_through_to_blocked(self):
        """Catches any future grade label the app does not know."""
        for stage_status, grade in _GRADE_FOR_STATUS.items():
            with self.subTest(stage_status):
                self.assertNotEqual(status_for_grade(grade), "blocked",
                                    f"{stage_status} -> {grade!r} is not an app grade")

    def test_every_stage_status_has_a_mapping(self):
        from agent_staged_contract import FINAL_STATUSES
        self.assertEqual(set(_GRADE_FOR_STATUS), set(FINAL_STATUSES))


class AgentNotesTests(unittest.TestCase):
    def test_notes_land_in_the_section_the_app_leaves_empty(self):
        report = render_report(title="t", front_matter={}, findings=[])
        self.assertIn("## Agent Notes\n\n## Decision Log", report)
        filled = agent_app_report.append_agent_notes(report, "- 관점 A가 지지")
        self.assertIn("## Agent Notes\n\n- 관점 A가 지지\n", filled)
        self.assertIn("## Decision Log", filled)

    def test_appending_nothing_leaves_the_report_untouched(self):
        report = render_report(title="t", front_matter={}, findings=[])
        self.assertEqual(agent_app_report.append_agent_notes(report, "   "), report)

    def test_the_cell_sections_are_never_disturbed_by_notes(self):
        result = agent_app_report.build_cell_result(EVIDENCE, sheet="CO(콜롬비아)")
        base = render_report(title="t", front_matter={}, findings=[render_finding(result)])
        filled = agent_app_report.append_agent_notes(base, "- 시트 일관성 의견 1건")
        self.assertEqual(base.split("## Agent Notes")[0], filled.split("## Agent Notes")[0])


if __name__ == "__main__":
    unittest.main()
