from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from report_manifest_adapter import approved_edits, normalize_report_manifest  # noqa: E402


class ReportManifestAdapterTests(unittest.TestCase):
    def _manifest(self, changes=None):
        return {
            "manifest_schema_version": 1,
            "report_id": "review-20260729-001",
            "source_file_id": "story-039",
            "changes": changes or [{
                "finding_id": "CO-C10", "sheet": "CO(콜롬비아)", "cell": "c10",
                "before": "Hola", "after": "Hola, ¿cómo estás?",
                "rule_ids": ["spanish-colombia-001"], "approval_status": "approved",
            }],
        }

    def test_extracts_only_approved_changes(self):
        raw = self._manifest(changes=[
            self._manifest()["changes"][0],
            {"finding_id": "CO-C11", "sheet": "CO(콜롬비아)", "cell": "C11", "before": "A", "after": "B", "approval_status": "rejected"},
        ])
        self.assertEqual(approved_edits(raw), [{
            "sheet": "CO(콜롬비아)", "cell": "C10", "before": "Hola", "after": "Hola, ¿cómo estás?",
        }])
        self.assertEqual(len(normalize_report_manifest(raw)["skipped"]), 1)

    def test_rejects_invalid_or_unapproved_manifests(self):
        with self.assertRaisesRegex(ValueError, "승인"):
            approved_edits(self._manifest(changes=[{
                "finding_id": "CO-C10", "sheet": "CO(콜롬비아)", "cell": "C10",
                "before": "A", "after": "B", "approval_status": "pending_approval",
            }]))
        with self.assertRaisesRegex(ValueError, "중복"):
            normalize_report_manifest(self._manifest(changes=[
                self._manifest()["changes"][0],
                {"finding_id": "CO-C10b", "sheet": "CO(콜롬비아)", "cell": "c10", "before": "A", "after": "B", "approval_status": "approved"},
            ]))
        with self.assertRaisesRegex(ValueError, "schema"):
            normalize_report_manifest({"manifest_schema_version": 2})


if __name__ == "__main__":
    unittest.main()
