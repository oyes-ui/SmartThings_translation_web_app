"""The published JSON Schemas must accept exactly what the runtime accepts.

A schema is the contract an out-of-process runner codes against. When the runtime
gained required run identity (run_id, executed_at) the schemas kept
additionalProperties: false without those fields, so a conforming runner was stuck:
omit them and the runtime refuses, include them and the schema refuses. These tests
fail on that class of drift rather than waiting for someone to hit it.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from agent_staged_contract import STAGES, validate_cell_review  # noqa: E402

SCHEMA_FOR_STAGE = {
    "cell_review": "cell_review.schema.json",
    "sheet_consistency_review": "sheet_review.schema.json",
    "lead_review": "lead_review.schema.json",
}
# Checked by validate_envelope for every stage.
RUNTIME_ENVELOPE_FIELDS = ("kind", "packet_id", "status", "stop_reason",
                           "model", "run_id", "executed_at")


def _schema(name: str) -> dict:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


class SchemaRuntimeConformanceTests(unittest.TestCase):
    def test_every_stage_schema_requires_the_runtime_envelope(self):
        for stage in STAGES:
            with self.subTest(stage):
                required = set(_schema(SCHEMA_FOR_STAGE[stage]).get("required", []))
                missing = [field for field in RUNTIME_ENVELOPE_FIELDS if field not in required]
                self.assertEqual(missing, [], f"{stage} schema is missing {missing}")

    def test_closed_schemas_still_define_every_field_they_require(self):
        """additionalProperties: false plus an undefined required field is unsatisfiable."""
        for stage in STAGES:
            with self.subTest(stage):
                schema = _schema(SCHEMA_FOR_STAGE[stage])
                if schema.get("additionalProperties") is not False:
                    continue
                undefined = set(schema.get("required", [])) - set(schema.get("properties", {}))
                self.assertEqual(undefined, set(), f"{stage}: required but undefined {undefined}")

    def test_the_cell_schema_requires_the_app_audit_evaluation(self):
        cell = _schema("cell_review.schema.json")["properties"]["cells"]["items"]
        self.assertIn("evaluation", cell["required"])
        self.assertEqual(cell["properties"]["evaluation"]["minItems"], 1)

    def test_a_schema_valid_cell_review_passes_the_runtime_validator(self):
        """The two contracts agree on one concrete payload, not just field-by-field."""
        packet = {"packet_id": "pkt", "audit_checklist": ["문법/유창성"],
                  "deterministic_evidence": [{"cell": "C7"}]}
        payload = {
            "kind": "cell_review", "packet_id": "pkt", "status": "completed",
            "stop_reason": "complete", "model": "m", "run_id": "r",
            "executed_at": "2026-08-12T00:00:00Z",
            "cells": [{"cell": "C7", "status": "pass", "after": None, "reason": "ok",
                       "evaluation": [{"category": "문법/유창성", "comment": "확인함"}],
                       "rule_ids": [], "used_prior_cell_context": False,
                       "prior_cell_refs": [], "prior_cell_influence": ""}],
        }
        schema = _schema("cell_review.schema.json")
        self.assertEqual(set(payload) - set(schema["properties"]), set())
        self.assertEqual(set(schema["required"]) - set(payload), set())
        validate_cell_review(payload, packet)


if __name__ == "__main__":
    unittest.main()
