from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from agent_review_contract import SPECIALIST_ROLES  # noqa: E402
from agent_role_prompts import ROLE_BRIEFS, build_role_prompt  # noqa: E402

PACKET = {
    "kind": "agent_sheet_review_packet",
    "packet_id": "pkt0000000000001",
    "workbook_name": "Story_001.xlsx",
    "target_sheet": "CO(콜롬비아)",
    "source_sheet": "US(미국)",
    "target_sections": [], "source_sections": [],
    "deterministic_evidence": [{"cell": "C7", "hard_rule_issues": ["[괄호 오류]"]}],
    "candidate_overlay": [],
    "semantic_rag_budget": 0,
    "review_mode": "multi_agent",
}


class AgentRolePromptsTests(unittest.TestCase):
    def test_every_specialist_role_has_a_brief(self):
        self.assertEqual(set(ROLE_BRIEFS), set(SPECIALIST_ROLES))

    def test_prompt_carries_packet_id_and_deterministic_evidence(self):
        prompt = build_role_prompt("grammar_fluency", PACKET)
        self.assertIn("pkt0000000000001", prompt)
        self.assertIn("[괄호 오류]", prompt)
        self.assertIn("CO(콜롬비아)", prompt)

    def test_prompt_states_the_role_boundary(self):
        prompt = build_role_prompt("semantic_fidelity", PACKET)
        self.assertIn("원문 의미 충실도", prompt)
        self.assertIn("네가 판단하지 않는 것", prompt)

    def test_prompt_never_contains_another_roles_opinion(self):
        """There is no parameter for other opinions, so anchoring cannot enter here."""
        polluted = {**PACKET, "opinions": [{"role": "semantic_fidelity", "after": "복사된 제안"}]}
        prompt = build_role_prompt("story_and_ui_coherence", polluted)
        self.assertNotIn("복사된 제안", prompt)
        self.assertIn("다른 관점의 의견을 참고하지 않는다", prompt)

    def test_hard_rule_internals_go_only_to_the_role_that_owns_them(self):
        packet = {**PACKET, "deterministic_evidence": [
            {"cell": "C7", "row_type": "title", "hard_rule_issues": ["[괄호 오류]"],
             "constraint_card": {"resolver": "internal"}},
        ]}
        owner = build_role_prompt("style_and_hard_rule_exceptions", packet)
        other = build_role_prompt("grammar_fluency", packet)
        self.assertIn("constraint_card", owner)
        self.assertNotIn("constraint_card", other)
        # The other role still learns that a hard rule already fired.
        self.assertIn("[괄호 오류]", other)

    def test_every_role_is_told_it_is_read_only(self):
        for role in SPECIALIST_ROLES:
            prompt = build_role_prompt(role, PACKET)
            self.assertIn("읽기 전용 검수자", prompt)
            self.assertIn("금지:", prompt)

    def test_unknown_role_is_rejected(self):
        with self.assertRaises(ValueError):
            build_role_prompt("proofreader", PACKET)

    def test_non_packet_input_is_rejected(self):
        with self.assertRaises(ValueError):
            build_role_prompt("grammar_fluency", {"target_sheet": "CO(콜롬비아)"})

    def test_unapproved_sheet_cannot_get_role_prompts(self):
        """The escalation is opt-in: a default packet must not spawn five roles."""
        with self.assertRaises(ValueError) as caught:
            build_role_prompt("grammar_fluency", {**PACKET, "review_mode": "lead_2pass"})
        self.assertIn("--multi-agent", str(caught.exception))

    def test_semantic_budget_is_surfaced(self):
        self.assertIn("**0회**", build_role_prompt("localization_tone", PACKET))
        self.assertIn("**3회**", build_role_prompt("localization_tone", {**PACKET, "semantic_rag_budget": 3}))


if __name__ == "__main__":
    unittest.main()
