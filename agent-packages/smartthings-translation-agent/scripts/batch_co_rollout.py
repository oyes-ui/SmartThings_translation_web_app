#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_co_rollout.py — 여러 워크북에 새 타겟 언어 시트를 순차로 추가·번역하는 오케스트레이터.

⚠️ Step 2에서 workbook_translate.py --pipeline 을 호출해 **LLM 크레딧을 소모**한다.
   실행 전 대상 파일 수를 확인하고, 필요하면 --prep-only 로 먼저 무료 사전 점검만 해본다.

원본(@translation_data/@excel 등)은 절대 수정하지 않는다. 모든 산출물은 --out-dir 아래에만
생성된다. 파일 간 처리는 항상 순차(한 번에 한 파일)다 — --max-concurrency 는
workbook_translate.py 내부의 셀 단위 동시성일 뿐 파일 간 병렬처리가 아니다.

각 파일마다:
  Step 0 (무료): 소스 시트 존재 / 신규 시트 부재 확인. 어긋나면 그 파일은 건너뛰고 사유 기록.
  Step 1 (무료): workbook_add_target_sheet.py 로 신규 시트 준비 → {stem}__prep.xlsx
  Step 2 (크레딧, --prep-only 시 생략): workbook_translate.py --pipeline 실행
  Step 3 (무료): 결과 재오픈 후 번역 셀 수를 기대치와 대조(0셀 성공 오보고 방지)
  Step 4: 매니페스트(manifest.json)에 파일마다 즉시 원자적으로 기록

사용 예:
  # 무료 사전 점검만 (파일 수·구조 확인, 크레딧 없음)
  python batch_co_rollout.py --input-dir "@translation_data/@excel" --out-dir ./co_batch_out --prep-only

  # 파일럿 (파일 목록 직접 지정, 크레딧 소모)
  python batch_co_rollout.py --files a.xlsx b.xlsx --out-dir ./co_batch_out \\
      --glossary runtime/glossary/latest_glossary_260803.csv --backtranslation-lang Korean
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import _app_pipeline as ap  # noqa: E402
import workbook_add_target_sheet as wats  # noqa: E402

# Statuses a resumed run may skip.  Errors and skips are always retried: they are
# usually transient or fixable, and freezing them would silently drop files.
#
# The set depends on what the current run is trying to produce.  ``prepped`` means
# the target sheet exists but nothing was translated, so a full run must still
# process it -- treating it as done would silently leave the file untranslated.
RESUMABLE_STATUSES = {"ok"}
PREP_RESUMABLE_STATUSES = {"ok", "prepped"}


def resumable_statuses(prep_only: bool) -> set[str]:
    return PREP_RESUMABLE_STATUSES if prep_only else RESUMABLE_STATUSES


def discover_files(files: list[str] | None, input_dir: str | None) -> list[Path]:
    if files:
        return [Path(f).expanduser() for f in files]
    if input_dir:
        base = Path(input_dir).expanduser()
        return sorted(
            p for p in base.glob("*.xlsx")
            if not p.name.startswith("~$")
        )
    raise ValueError("--files 또는 --input-dir 중 하나는 지정해야 합니다.")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_expected_cells(workbook: Path, source_sheet: str) -> int | None:
    """소스 시트에서 번역 대상이 될 셀 수(빈 값·'x' placeholder 제외)를 센다."""
    import openpyxl

    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        if source_sheet not in wb.sheetnames:
            return None
        ws = wb[source_sheet]
        count = 0
        for row in range(wats.CONTENT_ROW_START, wats.CONTENT_ROW_END + 1):
            val = ws.cell(row=row, column=wats.CONTENT_COL).value
            s = str(val).strip() if val is not None else ""
            if s and s.lower() != "x":
                count += 1
        return count
    finally:
        wb.close()


def count_written_cells(workbook: Path, new_sheet: str, column: int) -> int | None:
    import openpyxl

    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        if new_sheet not in wb.sheetnames:
            return None
        ws = wb[new_sheet]
        count = 0
        for row in range(wats.CONTENT_ROW_START, wats.CONTENT_ROW_END + 1):
            val = ws.cell(row=row, column=column).value
            s = str(val).strip() if val is not None else ""
            if s and s.lower() != "x":
                count += 1
        return count
    finally:
        wb.close()


def process_one(args, workbook: Path, out_dir: Path) -> dict:
    stem = workbook.stem
    entry = {
        "source": str(workbook),
        "status": "pending",
        "prep_path": None,
        "translated_path": None,
        "cells_expected": None,
        "cells_written": None,
        "backtranslation_cells_written": None,
        "errors": [],
        "started_at": now_iso(),
        "finished_at": None,
    }

    # Step 0: 사전 점검 (무료)
    sheet_names = ap.workbook_sheetnames(workbook)
    if args.source_sheet not in sheet_names:
        entry["status"] = "skipped"
        entry["errors"].append(f"소스 시트 없음: {args.source_sheet}")
        entry["finished_at"] = now_iso()
        return entry
    if args.template_sheet not in sheet_names:
        entry["status"] = "skipped"
        entry["errors"].append(f"템플릿 시트 없음: {args.template_sheet}")
        entry["finished_at"] = now_iso()
        return entry

    entry["cells_expected"] = count_expected_cells(workbook, args.source_sheet)

    # Step 1: 신규 시트(+역번역 시트) 준비 (무료)
    prep_path = out_dir / f"{stem}__prep.xlsx"
    result = wats.add_target_sheet(
        workbook, args.template_sheet, args.new_sheet, args.lang_code,
        backtranslation_sheet=args.backtranslation_sheet,
    )
    if result["status"] == "error":
        entry["status"] = "error"
        entry["errors"].append(result["reason"])
        entry["finished_at"] = now_iso()
        return entry
    import os
    tmp = prep_path.with_suffix(prep_path.suffix + ".tmp")
    result["wb"].save(tmp)
    os.replace(tmp, prep_path)
    entry["prep_path"] = str(prep_path)

    if args.prep_only:
        entry["status"] = "prepped"
        entry["finished_at"] = now_iso()
        return entry

    # Step 2: 실제 번역+검수 (LLM 크레딧)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "workbook_translate.py"), str(prep_path),
        "--pipeline", "--single-source",
        "--source-sheet", args.source_sheet,
        "--sheets", args.new_sheet,
        "--max-concurrency", str(args.max_concurrency),
        "--translation-model", args.translation_model,
        "--audit-model", args.audit_model,
        "--json",
    ]
    if args.glossary:
        cmd += ["--glossary", args.glossary]
    if args.backtranslation_lang:
        cmd += ["--backtranslation-lang", args.backtranslation_lang]
    if args.backtranslation_sheet:
        cmd += ["--backtranslation-sheet", args.backtranslation_sheet]
    if args.app_root:
        cmd += ["--app-root", args.app_root]

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    try:
        res = json.loads(proc.stdout)
    except json.JSONDecodeError:
        entry["status"] = "error"
        entry["errors"].append(
            f"workbook_translate.py 출력 파싱 실패 (returncode={proc.returncode}): "
            f"{proc.stdout[-2000:]} / stderr: {proc.stderr[-2000:]}"
        )
        entry["finished_at"] = now_iso()
        return entry

    if res.get("status") != "ok" or not res.get("excel_path"):
        entry["status"] = "error"
        entry["errors"].append(res.get("error") or res.get("errors") or "번역 실패(사유 불명)")
        entry["finished_at"] = now_iso()
        return entry

    translated_path = Path(res["excel_path"])
    entry["translated_path"] = str(translated_path)

    # Step 3: 검증 — status:ok 만 믿지 않고 실제 셀 수를 재확인한다.
    written = count_written_cells(translated_path, args.new_sheet, column=wats.CONTENT_COL)
    entry["cells_written"] = written
    if args.backtranslation_lang and args.backtranslation_sheet:
        entry["backtranslation_cells_written"] = count_written_cells(
            translated_path, args.backtranslation_sheet, column=wats.CONTENT_COL,
        )

    if written is None:
        entry["status"] = "error"
        entry["errors"].append(f"번역 결과 시트를 찾을 수 없음: {args.new_sheet}")
    elif entry["cells_expected"] is not None and written != entry["cells_expected"]:
        entry["status"] = "error"
        entry["errors"].append(
            f"셀 수 불일치: 기대 {entry['cells_expected']}개, 실제 {written}개 "
            "(status:ok 였지만 실제로는 일부/전체 미처리 가능성)"
        )
    else:
        entry["status"] = "ok"

    entry["finished_at"] = now_iso()
    return entry


def write_manifest_atomic(manifest_path: Path, manifest: dict) -> None:
    ap.write_text_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
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
    p.add_argument("--translation-model", default="gemini-3.6-flash")
    p.add_argument("--audit-model", default="gpt-5.2")
    p.add_argument("--max-concurrency", type=int, default=5)
    p.add_argument("--app-root", help="app repo 경로 명시")
    p.add_argument("--prep-only", action="store_true",
                    help="Step 0~1만 실행(시트 준비까지), LLM 호출 없음 — 크레딧 0")
    p.add_argument("--resume", action="store_true",
                    help="기존 manifest의 ok/prepped 파일은 건너뛴다. error/skipped는 다시 시도한다")
    args = p.parse_args()

    if args.backtranslation_lang and not args.backtranslation_sheet:
        args.backtranslation_sheet = f"{args.new_sheet} 역번역"

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = discover_files(args.files, args.input_dir)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(2)

    if not files:
        print("❌ 처리할 파일이 없습니다.")
        sys.exit(2)

    manifest_path = out_dir / "manifest.json"
    manifest = {"created_at": now_iso(), "prep_only": args.prep_only, "files": {}}
    done: dict[str, dict] = {}
    if args.resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Only successes are carried over.  Re-running must retry errors and
        # skips, otherwise a transient failure would be frozen into the batch.
        allowed = resumable_statuses(args.prep_only)
        done = {stem: entry for stem, entry in (previous.get("files") or {}).items()
                if entry.get("status") in allowed}
        manifest["files"].update(done)
        manifest["resumed_from"] = previous.get("created_at")
        print(f"↻ resume: 이미 완료된 {len(done)}개 파일은 건너뜁니다.")

    print(f"▶ {len(files)}개 파일 순차 처리 시작 (prep-only={args.prep_only})")
    for i, workbook in enumerate(files, 1):
        stem = workbook.stem
        print(f"  [{i}/{len(files)}] {workbook.name} ...", end=" ", flush=True)
        if stem in done:
            print(f"↷ 건너뜀 ({done[stem]['status']})")
            continue
        if not workbook.is_file():
            manifest["files"][stem] = {
                "source": str(workbook), "status": "error",
                "errors": ["파일을 찾을 수 없음"], "finished_at": now_iso(),
            }
            print("❌ 파일 없음")
        else:
            entry = process_one(args, workbook, out_dir)
            manifest["files"][stem] = entry
            print(entry["status"])
        write_manifest_atomic(manifest_path, manifest)

    write_manifest_atomic(manifest_path, manifest)
    statuses = [f["status"] for f in manifest["files"].values()]
    print("─" * 40)
    print(f"완료: ok={statuses.count('ok')} prepped={statuses.count('prepped')} "
          f"skipped={statuses.count('skipped')} error={statuses.count('error')}")
    print(f"매니페스트: {manifest_path}")


if __name__ == "__main__":
    main()
