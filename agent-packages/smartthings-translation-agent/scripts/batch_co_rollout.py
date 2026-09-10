#!/usr/bin/env python3
"""Legacy CO argument adapter for workbook_batch; new work uses generic job settings.

Regional defaults live only here. Paid failures are never silently retried.
Old manifest.json is historical input, not proof of a completed generic job.
"""
import argparse
import json
from pathlib import Path
import _app_pipeline as ap
from workbook_batch import prepare, advance, read

# Historical consumers use these labels; the generic workflow owns actual resume state.
RESUMABLE_STATUSES = {"ok"}
PREP_RESUMABLE_STATUSES = {"ok", "prepped"}

def resumable_statuses(prep_only):
    return PREP_RESUMABLE_STATUSES if prep_only else RESUMABLE_STATUSES

def discover_files(files, input_dir):
    if files:
        return [Path(f).expanduser() for f in files]
    if input_dir:
        return sorted(p for p in Path(input_dir).expanduser().glob("*.xlsx") if not p.name.startswith("~$"))
    raise ValueError("--files 또는 --input-dir 필요")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", nargs="+", help="처리할 .xlsx 파일 목록")
    p.add_argument("--input-dir", help="이 폴더의 모든 *.xlsx 처리 (~$ 잠금파일 제외)")
    p.add_argument("--out-dir", required=True, help="모든 산출물(prep/manifest) 저장 위치")
    p.add_argument("--template-sheet", default="ES(스페인)")
    p.add_argument("--new-sheet", default="CO(콜롬비아)")
    p.add_argument("--lang-code", default="es_CO")
    p.add_argument("--source-sheet", default="US(미국)")
    p.add_argument("--glossary", help="용어집 CSV 경로(기본 파일명이 더 이상 없으므로 명시 권장)")
    p.add_argument("--backtranslation-lang", help="예: 'Korean' — 역번역 참조 언어 강제 지정")
    p.add_argument("--backtranslation-sheet",
                    help="역번역을 담을 별도 시트명. --backtranslation-lang만 주고 이 값을 "
                         "생략하면 '{new-sheet} 역번역'을 자동으로 사용한다.")
    p.add_argument("--translation-model", default="gemini-3.8-flash")
    p.add_argument("--audit-model", default="gpt-5.2")
    p.add_argument("--max-concurrency", type=int, default=5)
    p.add_argument("--app-root", help="app repo 경로 명시")
    p.add_argument("--prep-only", action="store_true",
                    help="Step 0~1만 실행(시트 준비까지), LLM 호출 없음 — 크레딧 0")
    p.add_argument("--resume", action="store_true",
                    help="workflow.json의 기존 작업 재개. 유료 실패는 --retry-job으로 명시")
    p.add_argument("--pipeline", action="store_true", help="유료 초벌 실행 승인")
    p.add_argument("--glossary-approved", action="store_true", help="용어집 적용·제외안 확정")
    p.add_argument("--activation-manifest")
    p.add_argument("--obsidian-dir")
    p.add_argument("--with-api-audit", action="store_true")
    p.add_argument("--retry-job", action="append", default=[])
    args = p.parse_args()
    root = Path(args.out_dir).expanduser().resolve()
    path = root / "workflow.json"
    if args.resume and path.exists():
        result = advance(path, pipeline=args.pipeline, retry_jobs=args.retry_job)
    else:
        if args.resume and (root / "manifest.json").exists():
            raise ValueError("Legacy manifest: inspect completed outputs and start with those workbooks; no automatic paid replay")
        app_root = ap.bootstrap_project(args.app_root)
        glossary = args.glossary or str(app_root / "runtime/glossary/latest_glossary.csv")
        backtranslation = args.backtranslation_sheet or (f"{args.new_sheet} 역번역" if args.backtranslation_lang else None)
        jobs = [{"workbook": str(file.resolve()), "sheet": args.new_sheet, "source_sheet": args.source_sheet,
                 "glossary": glossary, "glossary_approved": args.glossary_approved,
                 "activation_manifest": args.activation_manifest,
                 "prepare": {"template_sheet": args.template_sheet, "lang_code": args.lang_code},
                 "prep_only": args.prep_only, "translate": not args.prep_only, "review": not args.prep_only,
                 "api_audit": args.with_api_audit or bool(backtranslation),
                 "backtranslation_sheet": backtranslation, "backtranslation_lang": args.backtranslation_lang,
                 "translation_model": args.translation_model, "audit_model": args.audit_model,
                 "max_concurrency": args.max_concurrency} for file in discover_files(args.files, args.input_dir)]
        prepare({"app_root": str(app_root), "obsidian_dir": args.obsidian_dir, "jobs": jobs}, root)
        result = advance(path, pipeline=args.pipeline)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
