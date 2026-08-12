from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from agent_stage_prompts import sheet_prompt  # noqa: E402


class SheetPromptTests(unittest.TestCase):
    def test_requires_non_glossary_lexical_consistency_sweep(self):
        packet = {
            "packet_id": "pkt-tv", "source_sections": [], "target_sections": [],
            "cell_snapshot": {"C7": "Control del televisor"},
            "deterministic_evidence": [{
                "cell": "C7", "source_text": "TV remote", "target_text": "Control del televisor",
                "constraint_card": {"terms": []},
            }],
            "activation_candidates": [],
        }
        cell = {
            "kind": "cell_review", "packet_id": "pkt-tv", "status": "completed",
            "stop_reason": "complete", "cells": [{
                "cell": "C7", "status": "pass", "after": None, "reason": "ok", "rule_ids": [],
                "used_prior_cell_context": False, "prior_cell_refs": [], "prior_cell_influence": "",
            }],
        }
        prompt = sheet_prompt(packet, cell)
        self.assertIn("용어집에 없는 반복 표기", prompt)
        self.assertIn("TV / televisor", prompt)
        self.assertIn("human_review_required", prompt)


if __name__ == "__main__":
    unittest.main()
