from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from agent_review_contract import (  # noqa: E402
    SPECIALIST_ROLES, SemanticRagBudget, detect_role_anchoring, merge_subjective_opinions,
)

PACKET = {"packet_id": "pkt0000000000001", "target_sheet": "CO(콜롬비아)", "cell_snapshot": {"C10": "Hola"}}


def _all_roles(base):
    """Opinions from every role so the sheet counts as complete."""
    filler = [{**base, "role": role, "finding_id": f"filler-{role}", "stance": "review", "after": None}
              for role in SPECIALIST_ROLES]
    return filler


class AgentReviewContractTests(unittest.TestCase):
    def test_semantic_budget_never_exceeds_approved_limit(self):
        budget = SemanticRagBudget(limit=2)
        results = []
        threads = [threading.Thread(target=lambda i=i: results.append(budget.reserve(f"e{i}"))) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(results), 2)
        self.assertEqual(budget.report()["semantic_used"], 2)

    def test_two_independent_roles_are_required_for_subjective_change(self):
        base = {"sheet": "CO(콜롬비아)", "cell": "C10", "finding_id": "co-c10", "after": "Buenas",
                "stance": "support", "constraint_status": "pass", "packet_id": PACKET["packet_id"]}
        merged = merge_subjective_opinions([
            {**base, "role": "grammar_fluency"},
            {**base, "role": "localization_tone"},
            *_all_roles(base),
        ], packet=PACKET)
        self.assertEqual(merged.sheet_status, "completed")
        self.assertEqual(len(merged.proposals), 1)
        self.assertEqual(merged.proposals[0]["supporting_roles"], ["grammar_fluency", "localization_tone"])

    def test_single_or_opposed_subjective_change_is_queued(self):
        base = {"sheet": "CO(콜롬비아)", "cell": "C10", "finding_id": "co-c10", "after": "Buenas",
                "constraint_status": "pass", "packet_id": PACKET["packet_id"]}
        merged = merge_subjective_opinions([
            {**base, "role": "grammar_fluency", "stance": "support"},
            {**base, "role": "semantic_fidelity", "stance": "oppose"},
            *_all_roles(base),
        ], packet=PACKET)
        self.assertEqual(merged.proposals, [])
        self.assertTrue(any(item["reason"] == "specialist_disagreement_or_insufficient_support"
                            for item in merged.human_review_queue))

    def test_blocked_constraint_cannot_be_merged_even_with_two_supporters(self):
        base = {"sheet": "CO(콜롬비아)", "cell": "C10", "finding_id": "co-c10", "after": "Buenas",
                "stance": "support", "constraint_status": "blocked", "packet_id": PACKET["packet_id"]}
        merged = merge_subjective_opinions([
            {**base, "role": "grammar_fluency"},
            {**base, "role": "localization_tone"},
            *_all_roles(base),
        ], packet=PACKET)
        self.assertEqual(merged.proposals, [])
        self.assertEqual(merged.human_review_queue[0]["reason"], "blocked_by_deterministic_constraint")

    def test_missing_role_makes_the_sheet_incomplete_and_emits_no_proposals(self):
        base = {"sheet": "CO(콜롬비아)", "cell": "C10", "finding_id": "co-c10", "after": "Buenas",
                "stance": "support", "constraint_status": "pass", "packet_id": PACKET["packet_id"]}
        merged = merge_subjective_opinions([
            {**base, "role": "grammar_fluency"},
            {**base, "role": "localization_tone"},
        ], packet=PACKET)
        self.assertEqual(merged.sheet_status, "incomplete")
        self.assertEqual(merged.proposals, [])
        self.assertIn("semantic_fidelity", merged.missing_roles)
        self.assertTrue(any(item["reason"] == "sheet_incomplete_missing_role_opinion"
                            for item in merged.human_review_queue))

    def test_opinion_answering_a_different_packet_is_not_counted(self):
        base = {"sheet": "CO(콜롬비아)", "cell": "C10", "finding_id": "co-c10", "after": "Buenas",
                "stance": "support", "constraint_status": "pass", "packet_id": PACKET["packet_id"]}
        opinions = _all_roles(base)
        opinions[0] = {**opinions[0], "packet_id": "stale-packet-id"}
        merged = merge_subjective_opinions(opinions, packet=PACKET)
        self.assertEqual(merged.sheet_status, "incomplete")
        self.assertEqual(merged.agent_runs[0]["status"], "packet_mismatch")

    def test_deterministic_proposals_bypass_the_two_role_gate(self):
        base = {"sheet": "CO(콜롬비아)", "cell": "C10", "finding_id": "f", "after": None,
                "stance": "review", "constraint_status": "pass", "packet_id": PACKET["packet_id"]}
        merged = merge_subjective_opinions(
            _all_roles(base), packet=PACKET,
            deterministic_proposals=[{"finding_id": "hard-1", "sheet": "CO(콜롬비아)", "cell": "C11",
                                      "after": "Buenas", "rule_ids": ["glossary"]}],
        )
        self.assertEqual(len(merged.proposals), 1)
        self.assertEqual(merged.proposals[0]["origin"], "deterministic_hard_rule")

    def test_role_echoing_another_role_is_reported_as_non_independent(self):
        """The ES_CO shape: one role's supports are a strict subset of another's."""
        common = {"sheet": "CO(콜롬비아)", "constraint_status": "pass", "stance": "support",
                  "packet_id": PACKET["packet_id"]}
        opinions = []
        for index, after in enumerate(["Modo cine", "hogar seguro"], start=1):
            shared = {**common, "cell": f"C1{index}", "finding_id": f"co-c1{index}", "after": after}
            opinions.append({**shared, "role": "semantic_fidelity"})
            opinions.append({**shared, "role": "story_and_ui_coherence", "reason": "다른 사유만 새로 씀"})
        # semantic_fidelity also found something on its own, as it did in ES_CO.
        opinions.append({**common, "role": "semantic_fidelity", "cell": "C20",
                         "finding_id": "co-c20", "after": "solo suyo"})
        flags = detect_role_anchoring(opinions)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["role"], "story_and_ui_coherence")
        self.assertEqual(flags[0]["echoes_role"], "semantic_fidelity")
        self.assertEqual(flags[0]["direction"], "subset")

    def test_independent_roles_are_not_flagged_as_anchored(self):
        common = {"sheet": "CO(콜롬비아)", "constraint_status": "pass", "stance": "support",
                  "packet_id": PACKET["packet_id"]}
        opinions = [
            {**common, "role": "semantic_fidelity", "cell": "C11", "finding_id": "a", "after": "X"},
            {**common, "role": "story_and_ui_coherence", "cell": "C11", "finding_id": "a", "after": "X"},
            {**common, "role": "story_and_ui_coherence", "cell": "C12", "finding_id": "b", "after": "Y"},
        ]
        self.assertEqual(detect_role_anchoring(opinions), [])


if __name__ == "__main__":
    unittest.main()
