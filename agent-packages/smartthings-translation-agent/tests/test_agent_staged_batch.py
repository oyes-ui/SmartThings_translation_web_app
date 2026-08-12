from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from agent_staged_batch import (advance_batch, atomic_json, job_key, prepare_batch,
                                refresh_manifest)


def packet(sheet: str, packet_id: str) -> dict:
    return {
        "packet_id": packet_id, "review_mode": "staged_cell_sheet_lead",
        "workbook_name": "story.xlsx", "target_sheet": sheet,
        "hard_constraint_preamble": "constraints win",
        "source_sections": [], "target_sections": [],
        "cell_snapshot": {"C7": "Actual"},
        "deterministic_evidence": [{
            "cell": "C7", "row_type": "title", "source_text": "Source",
            "target_text": "Actual", "hard_rule_issues": [],
            "constraint_card": {"terms": []}, "constraint_validation": {"status": "pass"},
        }],
        "activation_candidates": [],
    }


def cell_review(packet_id: str) -> dict:
    return {
        "kind": "cell_review", "packet_id": packet_id, "status": "completed",
        "stop_reason": "complete", "model": "test-model", "run_id": "cell-run", "executed_at": "2026-08-12T00:00:00Z", "cells": [{
            "cell": "C7", "status": "pass", "after": None, "reason": "ok",
            "rule_ids": [], "used_prior_cell_context": False,
            "prior_cell_refs": [], "prior_cell_influence": "",
        }],
    }


class StagedBatchTests(unittest.TestCase):
    def test_prepare_isolates_languages_and_honours_concurrency_limit(self):
        active = 0
        peak = 0

        async def fake_build(_workbook, sheet, **_kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return packet(sheet, f"pkt-{sheet}")

        with tempfile.TemporaryDirectory() as raw, patch("agent_staged_batch.build_packet", fake_build):
            root = Path(raw)
            manifest = asyncio.run(prepare_batch(
                root / "story.xlsx", ["DE(독일)", "FR(프랑스)", "JA(일본)"], root / "batch",
                root / "glossary.csv", root / "app", max_concurrency=2))
            self.assertEqual(peak, 2)
            self.assertEqual(manifest["status_counts"], {"awaiting_cell_review": 3})
            self.assertEqual(len({job["work_dir"] for job in manifest["jobs"]}), 3)
            for job in manifest["jobs"]:
                self.assertTrue(Path(job["work_dir"], "packet.json").is_file())
                self.assertTrue(Path(job["work_dir"], "cell_prompt.txt").is_file())

    def test_prepare_rejects_duplicate_language_jobs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaises(ValueError):
                asyncio.run(prepare_batch(
                    root / "story.xlsx", ["DE(독일)", "DE(독일)"], root / "batch",
                    root / "glossary.csv", root / "app", max_concurrency=2))

    def test_advance_gates_cell_before_exposing_sheet_prompt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            work = root / "jobs" / job_key(1, "DE(독일)")
            pkt = packet("DE(독일)", "pkt-de")
            atomic_json(work / "packet.json", pkt)
            atomic_json(work / "cell_review.json", cell_review("pkt-de"))
            manifest_path = root / "batch_manifest.json"
            atomic_json(manifest_path, {
                "schema_version": 1, "workbook": str(root / "story.xlsx"),
                "glossary": str(root / "glossary.csv"), "app_root": str(root / "app"),
                "max_concurrency": 3, "jobs": [{
                    "job_id": work.name, "sheet": "DE(독일)", "packet_id": "pkt-de",
                    "work_dir": str(work), "state": "cell_review_ready", "error": None,
                }],
            })
            passing = lambda cell, after: {
                "status": "pass", "blocked": False, "normalized": after,
                "violations": [], "review": [],
            }
            with patch("agent_staged_batch.resolver_validator", return_value=passing):
                result = advance_batch(manifest_path)
            self.assertEqual(result["status_counts"], {"awaiting_sheet_review": 1})
            self.assertTrue((work / "cell_review.validated.json").is_file())
            self.assertTrue((work / "sheet_prompt.txt").is_file())
            validated = json.loads((work / "cell_review.validated.json").read_text(encoding="utf-8"))
            self.assertEqual(validated["resolver_gate_status"], "completed")

    def test_error_is_visible_but_corrected_artifact_can_be_retried(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            work = root / "job"
            atomic_json(work / "packet.json", packet("DE(독일)", "pkt-de"))
            (work / "cell_review.json").write_text("{}", encoding="utf-8")
            manifest_path = root / "batch_manifest.json"
            base = {
                "workbook": str(root / "story.xlsx"), "glossary": str(root / "g.csv"),
                "app_root": str(root / "app"), "max_concurrency": 1,
                "jobs": [{"job_id": "job", "sheet": "DE(독일)", "packet_id": "pkt-de",
                          "work_dir": str(work), "state": "cell_review_ready", "error": None}],
            }
            atomic_json(manifest_path, base)
            with patch("agent_staged_batch.resolver_validator", return_value=lambda *_: None):
                failed = advance_batch(manifest_path)
            self.assertEqual(failed["status_counts"], {"error": 1})
            atomic_json(work / "cell_review.json", cell_review("pkt-de"))
            passing = lambda cell, after: {"status": "pass", "blocked": False,
                                           "normalized": after, "violations": [], "review": []}
            with patch("agent_staged_batch.resolver_validator", return_value=passing):
                retried = advance_batch(manifest_path)
            self.assertEqual(retried["status_counts"], {"awaiting_sheet_review": 1})
            self.assertIsNone(retried["jobs"][0]["error"])


if __name__ == "__main__":
    unittest.main()
