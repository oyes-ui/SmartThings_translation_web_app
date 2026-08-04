from __future__ import annotations
import sys, tempfile, unittest
from pathlib import Path
import openpyxl
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
from review_report_builder import build_review_artifacts
class ReviewReportBuilderTests(unittest.TestCase):
 def test_builds_pending_report_without_mutating_source(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/"s.xlsx"; wb=openpyxl.Workbook(); ws=wb.active; ws.title="CO(콜롬비아)"; ws["C10"]="Hola"; wb.save(path)
   manifest, report=build_review_artifacts(path,[{"sheet":"CO(콜롬비아)","cell":"C10","before":"Hola","after":"Buenas","rule_ids":["co-1"]}],report_id="review-1",source_file_id="story-1")
   self.assertEqual(manifest["changes"][0]["approval_status"],"pending_approval"); self.assertIn("#### 제안 번역문",report); self.assertEqual(openpyxl.load_workbook(path)["CO(콜롬비아)"]["C10"].value,"Hola")

 def test_accepts_empty_proposals_with_sheet_review_context(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/"s.xlsx"; wb=openpyxl.Workbook(); wb.active.title="CO(콜롬비아)"; wb.save(path)
   manifest, report=build_review_artifacts(path,[],report_id="review-empty",source_file_id="story-1",review_context={
    "sheet_reviews":[{"sheet":"CO(콜롬비아)","status":"completed"}],
    "agent_runs":[{"role":"grammar_fluency","status":"completed"}],
    "rag_usage":{"semantic_budget":2,"semantic_used":0},
    "human_review_queue":[],"deterministic_checks":[],
   })
   self.assertEqual(manifest["manifest_schema_version"],2); self.assertEqual(manifest["changes"],[])
   self.assertIn("제안 없음",report); self.assertIn("semantic: 0 / 2",report)
if __name__=="__main__": unittest.main()
