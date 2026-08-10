from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from excel_live_manifest import approved_edits, normalize_manifest  # noqa: E402


class ExcelLiveManifestTests(unittest.TestCase):
    def _manifest(self, **overrides):
        base = {
            "surface": "chatgpt_excel",
            "mode": "draft",
            "approval": "approved",
            "glossary": {"locale": "es_CO", "version": "v1", "checksum": "abc"},
            "changes": [{
                "sheet": "CO(콜롬비아)", "cell": "c10",
                "before": "Hola", "after": "Hola, ¿cómo estás?", "verification": "verified",
            }],
        }
        base.update(overrides)
        return base

    def test_normalizes_approved_draft_to_apply_edits(self):
        result = approved_edits(self._manifest())
        self.assertEqual(result, [{
            "sheet": "CO(콜롬비아)", "cell": "C10", "before": "Hola", "after": "Hola, ¿cómo estás?",
        }])

    def test_accepts_claude_excel_surface(self):
        result = approved_edits(self._manifest(surface="claude_excel"))
        self.assertEqual(result, [{
            "sheet": "CO(콜롬비아)", "cell": "C10", "before": "Hola", "after": "Hola, ¿cómo estás?",
        }])

    def test_rejects_unknown_surface(self):
        with self.assertRaisesRegex(ValueError, "지원하지 않는 surface"):
            normalize_manifest(self._manifest(surface="copilot_excel"))

    def test_preview_or_unverified_change_cannot_apply(self):
        with self.assertRaisesRegex(ValueError, "preview"):
            approved_edits(self._manifest(mode="preview"))
        invalid = self._manifest(changes=[{
            "sheet": "CO(콜롬비아)", "cell": "C10", "before": "Hola", "after": "Buenas",
            "verification": "fallback_delivery",
        }])
        with self.assertRaisesRegex(ValueError, "verified"):
            approved_edits(invalid)

    def test_rejects_paths_duplicate_cells_and_stale_approval(self):
        with self.assertRaisesRegex(ValueError, "경로"):
            normalize_manifest(self._manifest(source_path="/secret/workbook.xlsx"))
        duplicate = self._manifest(changes=[
            {"sheet": "CO(콜롬비아)", "cell": "C10", "before": "A", "after": "B"},
            {"sheet": "CO(콜롬비아)", "cell": "c10", "before": "B", "after": "C"},
        ])
        with self.assertRaisesRegex(ValueError, "중복"):
            normalize_manifest(duplicate)
        with self.assertRaisesRegex(ValueError, "승인되지"):
            approved_edits(self._manifest(approval="pending"))


if __name__ == "__main__":
    unittest.main()
