from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from agent_review_contract import SemanticRagBudget, merge_subjective_opinions  # noqa: E402


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
        base = {"sheet": "CO(콜롬비아)", "cell": "C10", "finding_id": "co-c10", "after": "Buenas", "stance": "support"}
        proposals, queue = merge_subjective_opinions([
            {**base, "role": "grammar_fluency"},
            {**base, "role": "localization_tone"},
        ])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(queue, [])

    def test_single_or_opposed_subjective_change_is_queued(self):
        base = {"sheet": "CO(콜롬비아)", "cell": "C10", "finding_id": "co-c10", "after": "Buenas"}
        proposals, queue = merge_subjective_opinions([
            {**base, "role": "grammar_fluency", "stance": "support"},
            {**base, "role": "semantic_fidelity", "stance": "oppose"},
        ])
        self.assertEqual(proposals, [])
        self.assertEqual(len(queue), 1)


if __name__ == "__main__":
    unittest.main()
