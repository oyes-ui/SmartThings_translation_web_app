from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from obsidian_workflow import (  # noqa: E402
    init_base,
    publish_report,
    search_vault,
    stage_report,
    sync_status,
)


REPORT = """---
report_schema_version: 1
report_id: review-001
workflow: agent_review
status: draft
source_file_id: story-001
generated_at: 2026-08-03T00:00:00+00:00
---

# 번역 검수 리포트

## 셀 검수

### CO(콜롬비아) · C10 {#co-c10}

```yaml
finding_id: CO-C10
apply_status: pending_approval
rule_ids:
  - co-tone-tu
```

#### 현재 번역문

```text
Hola
```

#### 제안 번역문

```text
Buenas
```

## CO 콜롬비아

CO old section

## BR 포르투갈어(브라질)

BR unchanged section

## Decision Log

- existing decision
"""


class ObsidianWorkflowTests(unittest.TestCase):
    def _report(self, directory: Path, text: str = REPORT) -> Path:
        path = directory / "review.md"
        path.write_text(text, encoding="utf-8")
        return path

    def test_stage_preserves_report_contract_and_adds_obsidian_elements(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            staged = directory / "staged.md"
            result = stage_report(self._report(directory), staged)
            text = staged.read_text(encoding="utf-8")
            self.assertEqual(result["status"], "ok")
            self.assertIn("report_id: review-001", text)
            self.assertIn("{#co-c10}", text)
            self.assertIn("apply_status: pending_approval", text)
            self.assertIn("co-tone-tu", text)
            self.assertIn("Hola", text)
            self.assertIn("Buenas", text)
            self.assertIn("translation-review", text)
            self.assertIn("[!note]", text)
            self.assertIn("- existing decision", text)

    def test_publish_requires_apply_and_preserves_unrelated_locale(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            vault = directory / "vault"
            vault.mkdir()
            staged = directory / "staged.md"
            stage_report(self._report(directory), staged)
            with self.assertRaises(PermissionError):
                publish_report(staged, vault, "Reports/review.md", apply=False)
            self.assertFalse((vault / "Reports/review.md").exists())

            existing = vault / "Reports/review.md"
            existing.parent.mkdir()
            existing.write_text(REPORT.replace("CO old section", "CO prior section"), encoding="utf-8")
            result = publish_report(staged, vault, "Reports/review.md", apply=True, locale="CO")
            text = existing.read_text(encoding="utf-8")
            self.assertEqual(result["operation"], "locale_updated")
            self.assertIn("CO old section", text)
            self.assertIn("BR unchanged section", text)
            self.assertIn("- existing decision", text)

    def test_search_falls_back_to_filesystem_when_no_vault_name(self):
        with tempfile.TemporaryDirectory() as raw:
            vault = Path(raw)
            (vault / "rule.md").write_text("콜롬비아 tú 기준", encoding="utf-8")
            result = search_vault(vault, "tú", limit=5)
            self.assertEqual(result["mode"], "filesystem_fallback")
            self.assertEqual(result["results"][0]["path"], "rule.md")

    def test_sync_requires_valid_result_manifest_then_marks_applied(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            report = self._report(directory)
            invalid = directory / "invalid.json"
            invalid.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                sync_status(report, invalid, directory / "no-write.md")
            self.assertFalse((directory / "no-write.md").exists())

            result_manifest = directory / "delivery.json"
            result_manifest.write_text(json.dumps({
                "status": "ok", "final": "/safe/final.xlsx", "glossary": "/safe/glossary.csv",
                "delivery_manifest": "/safe/delivery.json", "highlight_report": "/safe/highlight.txt",
            }), encoding="utf-8")
            output = directory / "applied.md"
            result = sync_status(report, result_manifest, output)
            text = output.read_text(encoding="utf-8")
            self.assertEqual(result["status"], "ok")
            self.assertIn("status: applied", text)
            self.assertIn("result_manifest: `delivery.json`", text)
            self.assertIn("highlight_report: `highlight.txt`", text)

    def test_init_base_requires_apply_and_creates_status_views(self):
        with tempfile.TemporaryDirectory() as raw:
            vault = Path(raw)
            with self.assertRaises(PermissionError):
                init_base(vault, "SmartThings Translation Reviews.base", apply=False)
            result = init_base(vault, "SmartThings Translation Reviews.base", apply=True)
            text = (vault / "SmartThings Translation Reviews.base").read_text(encoding="utf-8")
            self.assertEqual(result["operation"], "created")
            self.assertIn('project == "SmartThings Translation"', text)
            self.assertIn("Approval queue", text)
            self.assertIn("Applied", text)


if __name__ == "__main__":
    unittest.main()
