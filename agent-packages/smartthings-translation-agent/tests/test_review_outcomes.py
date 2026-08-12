from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from quality_scorecard import evaluate  # noqa: E402
from review_outcomes import (  # noqa: E402
    append_ledger, load_ledger, outcomes_from_manifest, scorecard_inputs, summarise,
)


def _change(finding_id="f1", after="Buenas", proposed=None, status="approved", **extra):
    return {
        "finding_id": finding_id, "sheet": "CO(콜롬비아)", "cell": "C11", "row_type": "description",
        "origin": "subjective_consensus", "supporting_roles": ["grammar_fluency", "localization_tone"],
        "before": "Hola", "after": after, "proposed_after": proposed if proposed is not None else after,
        "approval_status": status, "rule_ids": ["co-1"], **extra,
    }


def _manifest(changes, anchoring=None):
    return {"report_id": "r1", "packet_id": "pkt1", "changes": changes, "anchoring": anchoring or []}


class ReviewOutcomesTests(unittest.TestCase):
    def test_plain_approval_is_recorded_as_approved(self):
        entry = outcomes_from_manifest(_manifest([_change()]))[0]
        self.assertEqual(entry["decision"], "approved")
        self.assertEqual(entry["final_after"], "Buenas")

    def test_reviewer_rewriting_the_text_is_recorded_as_edited(self):
        """The agent's proposal survives the reviewer's edit, so it can be scored."""
        entry = outcomes_from_manifest(_manifest([_change(after="Qué más", proposed="Buenas")]))[0]
        self.assertEqual(entry["decision"], "edited")
        self.assertEqual(entry["proposed_after"], "Buenas")
        self.assertEqual(entry["final_after"], "Qué más")

    def test_rejection_keeps_the_cell_unchanged(self):
        entry = outcomes_from_manifest(_manifest([
            _change(status="rejected", rejection_reason="원문 의미가 바뀜")
        ]))[0]
        self.assertEqual(entry["decision"], "rejected")
        self.assertEqual(entry["final_after"], "Hola")
        self.assertEqual(entry["rejection_reason"], "원문 의미가 바뀜")

    def test_unreviewed_manifest_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            outcomes_from_manifest(_manifest([_change(status="pending_approval")]))
        self.assertIn("pending_approval", str(caught.exception))

    def test_a_mistyped_status_fails_instead_of_counting_as_a_rejection(self):
        """A typo must not be scored as 'the agent proposed something wrong'."""
        for bad in ("approvd", "", "hold", "accept"):
            with self.assertRaises(ValueError) as caught:
                outcomes_from_manifest(_manifest([_change(status=bad)]))
            self.assertIn("approval_status", str(caught.exception))

    def test_support_from_an_echoing_role_is_marked_suspect(self):
        manifest = _manifest([_change()], anchoring=[
            {"role": "localization_tone", "echoes_role": "grammar_fluency", "direction": "subset"},
        ])
        self.assertEqual(outcomes_from_manifest(manifest)[0]["independence"], "suspect")

    def test_suspect_evidence_is_excluded_from_the_scorecard(self):
        ledger = [
            {"finding_id": "a", "before": "A", "proposed_after": "B", "final_after": "B",
             "independence": "verified", "origin": "deterministic_hard_rule", "decision": "approved"},
            {"finding_id": "b", "before": "C", "proposed_after": "D", "final_after": "C",
             "independence": "suspect", "origin": "subjective_consensus", "decision": "rejected"},
        ]
        golden, results = scorecard_inputs(ledger)
        self.assertEqual([row["id"] for row in golden], ["a"])
        golden_all, _ = scorecard_inputs(ledger, verified_only=False)
        self.assertEqual(len(golden_all), 2)

    def test_ledger_round_trip_replaces_a_revised_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            append_ledger(path, outcomes_from_manifest(_manifest([_change()])))
            append_ledger(path, outcomes_from_manifest(_manifest([_change(status="rejected")])))
            ledger = load_ledger(path)
            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger[0]["decision"], "rejected")

    def test_summary_splits_counts_by_risk_tier(self):
        ledger = outcomes_from_manifest(_manifest([
            _change(finding_id="a", origin="deterministic_hard_rule"),
            _change(finding_id="b", origin="deterministic_hard_rule"),
            _change(finding_id="c", status="rejected"),
        ]))
        summary = summarise(ledger)
        self.assertEqual(summary["by_origin"]["deterministic_hard_rule"]["approved"], 2)
        self.assertEqual(summary["by_origin"]["deterministic_hard_rule"]["accepted_as_proposed_rate"], 1.0)
        self.assertEqual(summary["by_origin"]["subjective_consensus"]["rejected"], 1)

    def test_ledger_feeds_quality_scorecard_unchanged(self):
        """The whole point: a reviewed manifest becomes a scoreable golden set."""
        ledger = outcomes_from_manifest(_manifest([
            _change(finding_id="a"),                                   # agent was right
            _change(finding_id="b", status="rejected"),                # agent was wrong
            _change(finding_id="c", after="사람 문안", proposed="에이전트 문안"),  # partly right
        ]))
        golden, results = scorecard_inputs(ledger)
        report = evaluate(golden, results)
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["exact_matches"], 1)
        self.assertEqual(report["false_positive_changes"], 1)  # the rejected one


if __name__ == "__main__":
    unittest.main()
