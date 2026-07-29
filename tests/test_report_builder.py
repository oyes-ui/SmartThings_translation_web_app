# -*- coding: utf-8 -*-
"""Markdown review report format (docs/report_format_spec.md)."""

import json
import re
import unittest

import yaml

from translation_web_app.report_builder import (
    NO_SUGGESTION,
    build_front_matter,
    render_finding,
    render_report,
    status_for_grade,
)


def _res(**overrides):
    base = {
        "sheet_name": "DE(독일)",
        "cell_ref": "C15",
        "source": "Make home care easier with SmartThings.",
        "target": "Mache die Pflege deines Zuhauses einfacher.",
        "case_section": "별도 지적 사항 없음",
        "glossary_section": "준수",
        "rag_text": "유사 DE 사례 2건",
        "back_translation": "Make home care easier.",
        "ai_text": "Needs Revision",
        "rag_json": "[]",
        "ai_json": json.dumps(
            {"grade": "Needs Revision", "suggested_fix": "Mit SmartThings wird es einfacher."},
            ensure_ascii=False,
        ),
    }
    base.update(overrides)
    return base


def _finding_yaml(markdown):
    match = re.search(r"```yaml\n(.*?)\n```", markdown, re.DOTALL)
    return yaml.safe_load(match.group(1))


class GradeMappingTests(unittest.TestCase):
    def test_known_grades(self):
        self.assertEqual(status_for_grade("Excellent"), "pass")
        self.assertEqual(status_for_grade("Good"), "warning")
        self.assertEqual(status_for_grade("Needs Revision"), "needs_revision")

    def test_missing_grade_is_blocked_not_pass(self):
        """No verdict must never be rounded down to a passing result."""
        self.assertEqual(status_for_grade(""), "blocked")
        self.assertEqual(status_for_grade("무엇인가"), "blocked")


class FindingTests(unittest.TestCase):
    def test_required_sections_present(self):
        out = render_finding(_res())
        for heading in ("#### 원문", "#### 현재 번역문", "#### 제안 번역문",
                        "#### 검수 상세", "#### 원본 검수 Payload"):
            self.assertIn(heading, out)

    def test_metadata_block(self):
        meta = _finding_yaml(render_finding(_res()))
        self.assertEqual(meta["finding_id"], "DE-C15")
        self.assertEqual(meta["status"], "needs_revision")
        self.assertEqual(meta["apply_status"], "pending_approval")
        self.assertEqual(meta["sheet"], "DE(독일)")
        self.assertEqual(meta["cell"], "C15")

    def test_no_suggestion_marks_not_applicable(self):
        out = render_finding(_res(ai_json=json.dumps({"grade": "Excellent", "suggested_fix": ""})))
        self.assertEqual(_finding_yaml(out)["apply_status"], "not_applicable")
        self.assertIn(NO_SUGGESTION, out)

    def test_suggestion_identical_to_current_is_not_a_change(self):
        res = _res()
        out = render_finding(_res(ai_json=json.dumps(
            {"grade": "Good", "suggested_fix": res["target"]}, ensure_ascii=False)))
        self.assertEqual(_finding_yaml(out)["apply_status"], "not_applicable")

    def test_unparseable_audit_payload_does_not_raise(self):
        out = render_finding(_res(ai_json="not json at all"))
        self.assertEqual(_finding_yaml(out)["status"], "blocked")

    def test_backticks_in_content_do_not_break_the_fence(self):
        out = render_finding(_res(source="use ``` then ```` here"))
        # The opening fence must be longer than the longest run in the body.
        self.assertIn("`````text", out)


class ReportTests(unittest.TestCase):
    def _report(self, findings=None):
        return render_report(
            title="번역 검수 보고서",
            front_matter=build_front_matter(
                report_id="review-20260730-001",
                source_file_id="story-039.xlsx",
                translation_model="gemini-3.6-flash",
                audit_model="gpt-5.4-mini",
            ),
            findings=findings if findings is not None else [render_finding(_res())],
            summary_lines=["총 검수 항목: 1개"],
            usage_report="Gemini 100 tok",
        )

    def test_front_matter_is_valid_yaml_with_required_keys(self):
        report = self._report()
        match = re.match(r"^---\n(.*?)\n---\n", report, re.DOTALL)
        self.assertIsNotNone(match)
        meta = yaml.safe_load(match.group(1))
        for key in ("report_schema_version", "report_id", "workflow", "status",
                    "source_file_id", "generated_at"):
            self.assertIn(key, meta)
        self.assertEqual(meta["report_schema_version"], 1)

    def test_document_structure(self):
        report = self._report()
        for section in ("# 번역 검수 보고서", "## 요약", "## 셀 검수",
                        "## Agent Notes", "## Decision Log"):
            self.assertIn(section, report)

    def test_empty_report_is_still_well_formed(self):
        report = self._report(findings=[])
        self.assertIn("검수 항목이 없습니다.", report)
        self.assertIsNotNone(re.match(r"^---\n", report))

    def test_viewer_finding_regex_matches_emitted_output(self):
        """Contract: static/viewer/app.js collectFindings() must find every finding.

        Kept in sync by construction -- if the builder's heading or YAML block
        shape changes, this fails before the viewer silently shows zero findings.
        """
        report = self._report(findings=[render_finding(_res()),
                                        render_finding(_res(cell_ref="C16"))])
        viewer_re = re.compile(
            r"^###[ \t]+(.+?)[ \t]*(?:\{#([^}]+)\})?[ \t]*$\n+```yaml\n(.*?)\n```",
            re.MULTILINE | re.DOTALL,
        )
        matches = viewer_re.findall(report)
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(anchor for _, anchor, _ in matches))


if __name__ == "__main__":
    unittest.main()
