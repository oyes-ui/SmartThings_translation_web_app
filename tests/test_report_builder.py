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
        for marker in ("> [!quote] 원문", "> [!note] 현재 번역문", "> [!tip] 제안 번역문",
                       "#### 검수 상세", "#### 원본 검수 Payload"):
            self.assertIn(marker, out)

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

    def test_detail_items_are_individually_fenced_not_a_shared_table(self):
        """대소문자/용어집/RAG/역번역 are independent free-text reports; a shared
        table would flatten each one's own internal structure (e.g. glossary's
        own bullet sub-checks) into a single cell. Each becomes its own
        callout, blank-quoted so its fenced content nests correctly."""
        out = render_finding(_res(case_section="line1\nline2 | pipe"))
        self.assertIn(
            "> [!info] 대소문자 점검\n> ```text\n> line1\n> line2 | pipe\n> ```", out
        )
        self.assertNotIn("| 대소문자 점검 |", out)

    def test_callout_survives_a_blank_line_in_the_body(self):
        """A blank line inside fenced content must become a bare `>`, not a
        line that silently exits the blockquote."""
        out = render_finding(_res(case_section="line1\n\nline3"))
        self.assertIn("> line1\n>\n> line3", out)

    def test_ai_evaluation_renders_as_a_table_from_structured_json(self):
        """AI 검수 결과's category/comment pairs come from ai_json (the model's
        actual structured output), not by re-parsing the flattened ai_text."""
        out = render_finding(_res(ai_json=json.dumps({
            "grade": "Good",
            "evaluation": [
                {"category": "문법/유창성", "comment": "자연스럽습니다."},
                {"category": "현지화", "comment": "지역 어휘 | 표기 확인"},
            ],
            "suggested_fix": "",
        }, ensure_ascii=False)))
        self.assertIn("##### AI 검수 결과", out)
        self.assertIn("| 항목 | 결과 |", out)
        self.assertIn("| 문법/유창성 | 자연스럽습니다. |", out)
        self.assertIn("| 현지화 | 지역 어휘 \\| 표기 확인 |", out)

    def test_ai_text_falls_back_to_a_fence_when_no_evaluation_list(self):
        """Bypass/skip/error paths only ever set ai_text, not a structured
        evaluation list -- those must still render (as a fence), not vanish."""
        out = render_finding(_res(ai_text="[Bypassed: Translate Only Mode]",
                                   ai_json="{}"))
        self.assertIn("##### AI 검수 결과\n\n```text\n[Bypassed: Translate Only Mode]\n```", out)

    def test_payload_sections_collapse_by_default_in_raw_markdown(self):
        """Folding must live in the .md itself (native <details>, no `open`
        attribute) -- app.js's old client-side wrapping only ever affected the
        web viewer, never a report opened directly in Obsidian."""
        out = render_finding(_res())
        self.assertIn("#### 원본 검수 Payload\n\n<details>\n<summary>", out)
        self.assertIn("#### RAG Payload\n\n<details>\n<summary>", out)
        self.assertNotIn("<details open>", out)
        self.assertNotIn('<details open="', out)

    def test_rag_cases_render_as_separate_fences_not_merged(self):
        """Two RAG matches must be two independent fenced blocks inside the
        callout, not one fence with both cases concatenated -- each case has
        its own shape (type/score/story/section) that a shared blob loses."""
        rag_json = json.dumps([
            {"type": "semantic", "score": 92.3, "story_id": "story-039", "section": "C15",
             "source": "...", "target": "Ahorro inteligente de energía"},
            {"type": "exact", "score": 100.0, "story_id": "story-012", "section": "C7",
             "source": "...", "target": "Ahorro de energía inteligente"},
        ], ensure_ascii=False)
        out = render_finding(_res(rag_json=rag_json))
        self.assertIn("> [!example] RAG 일관성 참고", out)
        self.assertIn("> SEMANTIC (92.3%) | story-039 | C15", out)
        self.assertIn("> EXACT | story-012 | C7", out)
        # Two distinct fences, not one merged block: 4 fence markers (2 opens + 2 closes).
        rag_section = out[out.index("[!example] RAG"):out.index("[!quote] 역번역")]
        self.assertEqual(rag_section.count("```text"), 2)

    def test_rag_error_payload_is_not_mistaken_for_case_data(self):
        """rag_json can hold an `[{"error": ...}]` shape on lookup failure --
        that must fall back to rag_text, not be rendered as a fake case."""
        out = render_finding(_res(
            rag_text="RAG 조회 오류: timeout",
            rag_json=json.dumps([{"error": "timeout"}], ensure_ascii=False),
        ))
        self.assertIn("> RAG 조회 오류: timeout", out)


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
