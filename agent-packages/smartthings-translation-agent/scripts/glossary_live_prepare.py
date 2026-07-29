#!/usr/bin/env python3
"""Prepare a path-free glossary payload for a user-pasted Excel live import."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("csv_file");p.add_argument("--locale",required=True);p.add_argument("--version",required=True);p.add_argument("--term-column",default="target");p.add_argument("--output",required=True);a=p.parse_args()
 try:
  raw=Path(a.csv_file).read_bytes();rows=list(csv.DictReader(raw.decode("utf-8-sig").splitlines()));terms=[]
  for r in rows:
   v=(r.get(a.term_column) or "").strip()
   if v: terms.append(v)
  if not terms:raise ValueError("가져올 glossary term이 없습니다.")
  payload={"locale":a.locale,"version":a.version,"checksum":"sha256:"+hashlib.sha256(raw).hexdigest(),"terms":sorted(set(terms))}
  out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+".tmp");tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");os.replace(tmp,out);print(json.dumps({"status":"ok","terms":len(payload["terms"]),"payload":str(out)},ensure_ascii=False))
 except Exception as e:print(json.dumps({"status":"error","error":str(e)},ensure_ascii=False));sys.exit(2)
if __name__=="__main__":main()
