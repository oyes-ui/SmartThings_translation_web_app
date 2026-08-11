from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from batch_co_rollout import RESUMABLE_STATUSES  # noqa: E402


def _resumable(manifest: dict) -> dict:
    """Mirror of the resume filter applied in batch_co_rollout.main."""
    return {stem: entry for stem, entry in (manifest.get("files") or {}).items()
            if entry.get("status") in RESUMABLE_STATUSES}


class BatchResumeTests(unittest.TestCase):
    def test_only_successful_files_are_skipped_on_resume(self):
        manifest = {"files": {
            "story_001": {"status": "ok"},
            "story_002": {"status": "prepped"},
            "story_003": {"status": "error"},
            "story_004": {"status": "skipped"},
        }}
        self.assertEqual(sorted(_resumable(manifest)), ["story_001", "story_002"])

    def test_errors_are_retried_rather_than_frozen(self):
        """A transient failure must not become permanent just because it was logged."""
        self.assertNotIn("error", RESUMABLE_STATUSES)
        self.assertNotIn("skipped", RESUMABLE_STATUSES)

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
