#!/usr/bin/env python3
"""Compare agent/app proposals with a human-approved golden set."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
def load(v):
 p=Path(v).expanduser();return json.loads(p.read_text(encoding="utf-8") if p.is_file() else v)
def evaluate(golden,results):
 g={str(x["id"]):x for x in golden}; r={str(x["id"]):x for x in results}
 if set(g)!=set(r): raise ValueError("golden/results id 집합이 일치하지 않습니다.")
 total=len(g); exact=should=bad=0; rows=[]
 for k in sorted(g):
  x,y=g[k],r[k]; expected=x.get("expected_after",x.get("before")); proposed=y.get("proposed_after",y.get("before")); change_expected=expected!=x.get("before"); change_proposed=proposed!=y.get("before")
  match=proposed==expected; exact+=match; should+=change_expected; bad+=change_proposed and not change_expected
  rows.append({"id":k,"match":match,"expected_after":expected,"proposed_after":proposed,"false_positive":change_proposed and not change_expected,"missed_change":change_expected and not change_proposed})
 return {"total":total,"exact_matches":exact,"exact_match_rate":exact/total if total else 0,"expected_changes":should,"false_positive_changes":bad,"false_positive_rate":bad/total if total else 0,"rows":rows}
def main():
 p=argparse.ArgumentParser();p.add_argument("golden");p.add_argument("results");p.add_argument("--output");a=p.parse_args()
 try:
  out=evaluate(load(a.golden),load(a.results));text=json.dumps(out,ensure_ascii=False,indent=2)+"\n"
  if a.output:
   q=Path(a.output);q.parent.mkdir(parents=True,exist_ok=True);t=q.with_suffix(q.suffix+".tmp");t.write_text(text,encoding="utf-8");os.replace(t,q)
  print(text,end="")
 except Exception as e: print(json.dumps({"status":"error","error":str(e)},ensure_ascii=False));sys.exit(2)
if __name__=="__main__":main()
