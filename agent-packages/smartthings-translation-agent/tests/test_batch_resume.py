from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from batch_co_rollout import (  # noqa: E402
    PREP_RESUMABLE_STATUSES, RESUMABLE_STATUSES, resumable_statuses,
)

MIXED = {"files": {
    "story_001": {"status": "ok"},
    "story_002": {"status": "prepped"},
    "story_003": {"status": "error"},
    "story_004": {"status": "skipped"},
}}


def _resumable(manifest: dict, prep_only: bool = False) -> dict:
    """Mirror of the resume filter applied in batch_co_rollout.main."""
    allowed = resumable_statuses(prep_only)
    return {stem: entry for stem, entry in (manifest.get("files") or {}).items()
            if entry.get("status") in allowed}


class BatchResumeTests(unittest.TestCase):
    def test_a_full_run_skips_only_translated_files(self):
        self.assertEqual(sorted(_resumable(MIXED)), ["story_001"])

    def test_prepped_files_are_still_translated_on_a_full_run(self):
        """--prep-only leaves the sheet empty, so a real run must not skip it."""
        self.assertNotIn("prepped", resumable_statuses(prep_only=False))
        self.assertNotIn("story_002", _resumable(MIXED, prep_only=False))

    def test_a_prep_only_run_skips_files_already_prepared(self):
        self.assertEqual(sorted(_resumable(MIXED, prep_only=True)), ["story_001", "story_002"])

    def test_errors_are_retried_rather_than_frozen(self):
        """A transient failure must not become permanent just because it was logged."""
        for statuses in (RESUMABLE_STATUSES, PREP_RESUMABLE_STATUSES):
            self.assertNotIn("error", statuses)
            self.assertNotIn("skipped", statuses)

    def test_resume_of_an_empty_or_missing_manifest_skips_nothing(self):
        self.assertEqual(_resumable({}), {})
        self.assertEqual(_resumable({"files": {}}), {})

    def test_manifest_round_trips_as_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            payload = {"created_at": "t", "files": {"a": {"status": "ok"}, "b": {"status": "error"}}}
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(sorted(_resumable(json.loads(path.read_text(encoding="utf-8")))), ["a"])


if __name__ == "__main__":
    unittest.main()
