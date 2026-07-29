#!/usr/bin/env python3
"""Build a read-only review report and pending approval manifest."""
from __future__ import annotations
import argparse, json, os, re, sys
from datetime import datetime, timezone
from pathlib import Path
import openpyxl
CELL=re.compile(r"^[A-Z]{1,3}[1-9][0-9]*$")
def load_json(v):
 p=Path(v).expanduser(); return json.loads(p.read_text(encoding="utf-8") if p.is_file() else v)
def atomic(p,s):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".tmp"); t.write_text(s,encoding="utf-8"); os.replace(t,p)
def build_review_artifacts(workbook, proposals, *, report_id, source_file_id):
 if not Path(workbook).is_file(): raise FileNotFoundError("workbook을 찾을 수 없습니다.")
 if not isinstance(proposals,list) or not proposals: raise ValueError("proposals는 비어 있지 않은 list여야 합니다.")
 wb=openpyxl.load_workbook(workbook,read_only=True,data_only=False); changes=[]; seen=set()
 for i,x in enumerate(proposals):
  if not isinstance(x,dict): raise ValueError(f"proposals[{i}]는 object여야 합니다.")
  sheet=str(x.get("sheet","")).strip(); cell=str(x.get("cell","")).strip().upper()
  if sheet not in wb.sheetnames or not CELL.fullmatch(cell): raise ValueError(f"proposals[{i}] 대상이 올바르지 않습니다.")
  if (sheet,cell) in seen: raise ValueError(f"중복 제안: {sheet}!{cell}")
  seen.add((sheet,cell)); before=x.get("before"); after=x.get("after"); rules=x.get("rule_ids",[])
  if before != wb[sheet][cell].value: raise ValueError(f"proposals[{i}] before 불일치: {sheet}!{cell}")
  if not isinstance(after,str) or not after.strip() or after==before: raise ValueError(f"proposals[{i}].after가 올바르지 않습니다.")
  if not isinstance(rules,list) or not all(isinstance(r,str) and r for r in rules): raise ValueError(f"proposals[{i}].rule_ids는 문자열 list여야 합니다.")
  changes.append({"finding_id":x.get("finding_id") or f"{sheet.split('(')[0]}-{cell}","sheet":sheet,"cell":cell,"before":before,"after":after,"rule_ids":rules,"approval_status":"pending_approval","reason":str(x.get("reason", ""))})
 wb.close(); manifest={"manifest_schema_version":1,"report_id":report_id,"source_file_id":source_file_id,"changes":changes}
 now=datetime.now(timezone.utc).isoformat(); blocks=[]
 for c in changes:
  rules="\n".join(f"  - {r}" for r in c["rule_ids"]) or "  - review-pending"
  blocks.append(f"### {c['sheet']} · {c['cell']} {{#{c['finding_id'].lower()}}}\n\n```yaml\nfinding_id: {c['finding_id']}\nstatus: needs_revision\napply_status: pending_approval\nrule_ids:\n{rules}\n```\n\n#### 현재 번역문\n\n```text\n{c['before']}\n```\n\n#### 제안 번역문\n\n```text\n{c['after']}\n```\n\n#### 변경 이유\n\n{c['reason'] or '- 규칙/RAG 검토 후 사람 승인 대기'}\n")
 md=f"---\nreport_schema_version: 1\nreport_id: {report_id}\nworkflow: agent_review\nstatus: draft\nsource_file_id: {source_file_id}\ngenerated_at: {now}\n---\n\n# 번역 검수 리포트\n\n## 요약\n\n- 수정 제안: {len(changes)}건\n- 적용 상태: 모두 `pending_approval`\n\n## 셀 검수\n\n"+"\n".join(blocks)
 return manifest,md
def main():
 p=argparse.ArgumentParser(); p.add_argument("workbook");p.add_argument("proposals");p.add_argument("--report-id",required=True);p.add_argument("--source-file-id",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--json",action="store_true");a=p.parse_args()
 try:
  m,md=build_review_artifacts(Path(a.workbook).expanduser(),load_json(a.proposals),report_id=a.report_id,source_file_id=a.source_file_id);out=Path(a.output_dir).expanduser();rp=out/f"{a.report_id}.md";mp=out/f"{a.report_id}.manifest.json";atomic(rp,md);atomic(mp,json.dumps(m,ensure_ascii=False,indent=2)+"\n");r={"status":"ok","report":str(rp),"manifest":str(mp),"findings":len(m["changes"])};print(json.dumps(r,ensure_ascii=False,indent=2) if a.json else f"✅ {rp}")
 except Exception as e: print(json.dumps({"status":"error","error":str(e)},ensure_ascii=False));sys.exit(2)
if __name__=="__main__": main()
