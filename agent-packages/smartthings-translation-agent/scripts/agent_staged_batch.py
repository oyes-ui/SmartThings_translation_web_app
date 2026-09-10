#!/usr/bin/env python3
"""Coordinate independent language-sheet review pipelines without calling an LLM.

The batch is a durable hand-off boundary for an external agent runner. Each
language gets an isolated directory and may run in parallel with other
languages, while its cell -> sheet -> lead dependency remains strictly serial.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_sheet_merge import resolver_validator
from agent_sheet_review import build_packet
from agent_stage_prompts import cell_prompt, lead_prompt, sheet_prompt
from agent_staged_contract import gate_stage, merge_staged_reviews
from review_report_builder import build_review_artifacts, write_artifacts


SCHEMA_VERSION = 1
RAW_CELL = "cell_review.json"
VALID_CELL = "cell_review.validated.json"
RAW_SHEET = "sheet_review.json"
VALID_SHEET = "sheet_review.validated.json"
RAW_LEAD = "lead_review.json"
FINAL_SUMMARY = "final_summary.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object가 필요합니다: {path}")
    return value


def job_key(index: int, sheet: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "-", sheet).strip("-").lower() or "sheet"
    digest = hashlib.sha256(sheet.encode("utf-8")).hexdigest()[:8]
    return f"{index:03d}-{slug}-{digest}"


def _paths(job: dict[str, Any]) -> dict[str, Path]:
    root = Path(job["work_dir"])
    return {
        "root": root,
        "packet": root / "packet.json",
        "cell_prompt": root / "cell_prompt.txt",
        "cell_raw": root / RAW_CELL,
        "cell_valid": root / VALID_CELL,
        "sheet_prompt": root / "sheet_prompt.txt",
        "sheet_raw": root / RAW_SHEET,
        "sheet_valid": root / VALID_SHEET,
        "lead_prompt": root / "lead_prompt.txt",
        "lead_raw": root / RAW_LEAD,
        "summary": root / FINAL_SUMMARY,
        "output": root / "output",
    }


def derive_state(job: dict[str, Any]) -> str:
    paths = _paths(job)
    if paths["summary"].is_file():
        summary = read_json(paths["summary"])
        return ("completed" if summary.get("sheet_status") in {None, "completed", "complete"}
                else "incomplete") if summary.get("status") == "ok" else "error"
    if paths["lead_raw"].is_file():
        return "lead_review_ready"
    if paths["lead_prompt"].is_file():
        return "awaiting_lead_review"
    if paths["sheet_raw"].is_file():
        return "sheet_review_ready"
    if paths["sheet_prompt"].is_file():
        return "awaiting_sheet_review"
    if paths["cell_raw"].is_file():
        return "cell_review_ready"
    if paths["cell_prompt"].is_file():
        return "awaiting_cell_review"
    return "preparing"


def refresh_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    ready: list[dict[str, str]] = []
    for job in manifest.get("jobs", []):
        state = "error" if job.get("error") else derive_state(job)
        job["state"] = state
        counts[state] = counts.get(state, 0) + 1
        prompt_name = {
            "awaiting_cell_review": "cell_prompt.txt",
            "awaiting_sheet_review": "sheet_prompt.txt",
            "awaiting_lead_review": "lead_prompt.txt",
        }.get(state)
        if prompt_name:
            ready.append({"job_id": job["job_id"], "sheet": job["sheet"],
                          "stage": state.removeprefix("awaiting_").removesuffix("_review"),
                          "prompt": str(Path(job["work_dir"]) / prompt_name)})
    manifest["status_counts"] = counts
    manifest["ready_for_agent"] = ready
    manifest["updated_at"] = now_iso()
    return manifest


async def prepare_batch(workbook: Path, sheets: list[str], work_dir: Path, glossary: Path,
                        app_root: Path, *, max_concurrency: int,
                        semantic_rag_budget: int = 0,
                        activation_manifest: Path | None = None,
                        candidate_overlay: Path | None = None,
                        source_sheet: str | None = None, sheet_langs: dict | None = None) -> dict[str, Any]:
    if not sheets or len(sheets) != len(set(sheets)):
        raise ValueError("--sheets에는 중복 없는 시트를 하나 이상 지정하세요.")
    if max_concurrency < 1:
        raise ValueError("--max-concurrency는 1 이상이어야 합니다.")
    work_dir = work_dir.expanduser().resolve()
    manifest_path = work_dir / "batch_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("기존 batch_manifest.json이 있습니다. status/advance로 재개하세요.")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def prepare_one(index: int, sheet: str) -> dict[str, Any]:
        key = job_key(index, sheet)
        root = work_dir / "jobs" / key
        async with semaphore:
            packet = await build_packet(
                workbook, sheet, glossary=glossary, app_root=app_root,
                semantic_rag_budget=semantic_rag_budget,
                activation_manifest=activation_manifest,
                candidate_overlay=candidate_overlay, source_sheet=source_sheet, sheet_langs=sheet_langs)
        atomic_json(root / "packet.json", packet)
        atomic_text(root / "cell_prompt.txt", cell_prompt(packet))
        return {"job_id": key, "sheet": sheet, "packet_id": packet["packet_id"],
                "work_dir": str(root), "state": "awaiting_cell_review", "error": None}

    results = await asyncio.gather(
        *(prepare_one(index, sheet) for index, sheet in enumerate(sheets, 1)),
        return_exceptions=True)
    jobs = []
    for index, (sheet, result) in enumerate(zip(sheets, results), 1):
        if isinstance(result, Exception):
            key = job_key(index, sheet)
            jobs.append({"job_id": key, "sheet": sheet, "packet_id": None,
                         "work_dir": str(work_dir / "jobs" / key), "state": "error",
                         "error": str(result)})
        else:
            jobs.append(result)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "agent_staged_language_batch",
        "created_at": now_iso(),
        "workbook": str(workbook.expanduser().resolve()),
        "glossary": str(glossary.expanduser().resolve()),
        "app_root": str(app_root.expanduser().resolve()),
        "activation_manifest": str(activation_manifest.expanduser().resolve()) if activation_manifest else None,
        "candidate_overlay": str(candidate_overlay.expanduser().resolve()) if candidate_overlay else None,
        "semantic_rag_budget_per_sheet": semantic_rag_budget,
        "max_concurrency": max_concurrency,
        "execution_policy": "parallel_across_languages; serial_cell_sheet_lead_within_language",
        "jobs": jobs,
    }
    refresh_manifest(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


def advance_job(job: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(job)
    packet = read_json(paths["packet"])
    validate = resolver_validator(packet, Path(manifest["glossary"]), Path(manifest["app_root"]), job["sheet"])
    if paths["cell_raw"].is_file() and not paths["cell_valid"].is_file():
        cell = gate_stage(read_json(paths["cell_raw"]), "cell_review", packet, validate)
        atomic_json(paths["cell_valid"], cell)
        atomic_text(paths["sheet_prompt"], sheet_prompt(packet, cell))
    if paths["sheet_raw"].is_file() and not paths["sheet_valid"].is_file():
        if not paths["cell_valid"].is_file():
            raise ValueError("sheet 결과보다 먼저 cell resolver 게이트가 완료되어야 합니다.")
        cell = read_json(paths["cell_valid"])
        sheet = gate_stage(read_json(paths["sheet_raw"]), "sheet_consistency_review", packet, validate)
        atomic_json(paths["sheet_valid"], sheet)
        atomic_text(paths["lead_prompt"], lead_prompt(packet, cell, sheet))
    if paths["lead_raw"].is_file() and not paths["summary"].is_file():
        if not paths["cell_valid"].is_file() or not paths["sheet_valid"].is_file():
            raise ValueError("lead 결과보다 먼저 cell/sheet resolver 게이트가 완료되어야 합니다.")
        cell, sheet = read_json(paths["cell_valid"]), read_json(paths["sheet_valid"])
        merged = merge_staged_reviews(packet, cell, sheet, read_json(paths["lead_raw"]), validate)
        report_id = f"{Path(manifest['workbook']).stem}-{job['job_id']}"
        report_manifest, markdown = build_review_artifacts(
            Path(manifest["workbook"]), merged, report_id=report_id,
            source_file_id=packet.get("workbook_name", ""),
            app_root=Path(manifest["app_root"]))
        outputs = write_artifacts(report_manifest, markdown, paths["output"], report_id)
        atomic_json(paths["summary"], {
            "status": "ok", "sheet_status": merged.sheet_status,
            "changes": len(report_manifest["changes"]),
            "human_review_queue": len(report_manifest["review_context"]["human_review_queue"]),
            "resolver_gate": merged.resolver_gate,
            "anchoring_metrics": merged.anchoring_metrics, **outputs,
        })
    job["error"] = None
    job["state"] = derive_state(job)
    return job


def advance_batch(manifest_path: Path, max_concurrency: int | None = None) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    workers = max_concurrency or int(manifest.get("max_concurrency", 3))
    if workers < 1:
        raise ValueError("--max-concurrency는 1 이상이어야 합니다.")
    candidates = [job for job in manifest.get("jobs", [])
                  if derive_state(job) in {"cell_review_ready", "sheet_review_ready", "lead_review_ready"}]
    by_id = {job["job_id"]: job for job in manifest.get("jobs", [])}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(advance_job, dict(job), manifest): job for job in candidates}
        for future in as_completed(futures):
            original = futures[future]
            try:
                by_id[original["job_id"]] = future.result()
            except Exception as error:
                failed = dict(original)
                failed["state"], failed["error"] = "error", str(error)
                by_id[original["job_id"]] = failed
    manifest["jobs"] = [by_id[job["job_id"]] for job in manifest.get("jobs", [])]
    refresh_manifest(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


def print_summary(manifest: dict[str, Any], manifest_path: Path) -> None:
    print(json.dumps({"status": "ok", "manifest": str(manifest_path),
                      "max_concurrency": manifest.get("max_concurrency"),
                      "execution_policy": manifest.get("execution_policy"),
                      "status_counts": manifest.get("status_counts", {}),
                      "ready_for_agent": manifest.get("ready_for_agent", []),
                      "errors": [{"job_id": j["job_id"], "sheet": j["sheet"], "error": j.get("error")}
                                 for j in manifest.get("jobs", []) if j.get("error")]},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="언어별 packet/cell prompt를 제한 병렬 생성")
    prepare.add_argument("workbook", type=Path)
    prepare.add_argument("--sheets", nargs="+", required=True)
    prepare.add_argument("--work-dir", required=True, type=Path)
    prepare.add_argument("--glossary", required=True, type=Path)
    prepare.add_argument("--app-root", required=True, type=Path)
    prepare.add_argument("--source-sheet")
    prepare.add_argument("--sheet-langs", type=Path)
    prepare.add_argument("--activation-manifest", type=Path)
    prepare.add_argument("--candidate-overlay", type=Path)
    prepare.add_argument("--semantic-rag-budget", type=int, default=0)
    prepare.add_argument("--max-concurrency", type=int, default=3)
    for name in ("advance", "status"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", required=True, type=Path)
        if name == "advance":
            command.add_argument("--max-concurrency", type=int)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            manifest_path = args.work_dir.expanduser().resolve() / "batch_manifest.json"
            manifest = asyncio.run(prepare_batch(
                args.workbook, args.sheets, args.work_dir, args.glossary, args.app_root,
                max_concurrency=args.max_concurrency,
                semantic_rag_budget=args.semantic_rag_budget,
                activation_manifest=args.activation_manifest,
                candidate_overlay=args.candidate_overlay, source_sheet=args.source_sheet,
                sheet_langs=read_json(args.sheet_langs) if args.sheet_langs else None))
        elif args.command == "advance":
            manifest_path = args.manifest.expanduser().resolve()
            manifest = advance_batch(manifest_path, args.max_concurrency)
        else:
            manifest_path = args.manifest.expanduser().resolve()
            manifest = refresh_manifest(read_json(manifest_path))
            atomic_json(manifest_path, manifest)
        print_summary(manifest, manifest_path)
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        sys.exit(2)


if __name__ == "__main__":
    main()
