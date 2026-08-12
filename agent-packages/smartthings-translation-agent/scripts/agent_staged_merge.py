#!/usr/bin/env python3
"""Merge the three default review stages and emit the final report/manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_sheet_merge import resolver_validator
from agent_staged_contract import STAGES, incomplete_staged_result, merge_staged_reviews
from review_report_builder import build_review_artifacts, write_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--cell-review", required=True, type=Path)
    parser.add_argument("--sheet-review", required=True, type=Path)
    parser.add_argument("--lead-review", required=True, type=Path)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--glossary", required=True, type=Path)
    parser.add_argument("--app-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        if packet.get("review_mode") != "staged_cell_sheet_lead":
            raise ValueError("기본 3단계 패킷이 아닙니다.")
        stage_paths = dict(zip(STAGES, (args.cell_review, args.sheet_review, args.lead_review)))
        missing = [stage for stage, path in stage_paths.items() if not path.is_file()]
        if missing:
            merged = incomplete_staged_result(packet, missing)
        else:
            cell = json.loads(args.cell_review.read_text(encoding="utf-8"))
            sheet = json.loads(args.sheet_review.read_text(encoding="utf-8"))
            lead = json.loads(args.lead_review.read_text(encoding="utf-8"))
            if any(payload.get("resolver_gate_status") != "completed" for payload in (cell, sheet)):
                raise ValueError("cell/sheet 결과를 agent_stage_gate.py로 먼저 검증하세요.")
            validate = resolver_validator(packet, args.glossary.expanduser(), args.app_root.expanduser(),
                                          str(packet.get("target_sheet", "")))
            merged = merge_staged_reviews(packet, cell, sheet, lead, validate)
        manifest, markdown = build_review_artifacts(
            args.workbook.expanduser(), merged, report_id=args.report_id,
            source_file_id=packet.get("workbook_name", args.workbook.name),
            app_root=args.app_root)
        paths = write_artifacts(manifest, markdown, args.output_dir, args.report_id)
        print(json.dumps({"status": "ok", **paths, "sheet_status": merged.sheet_status,
                          "changes": len(manifest["changes"]),
                          "human_review_queue": len(manifest["review_context"]["human_review_queue"]),
                          "resolver_gate": merged.resolver_gate,
                          "anchoring_metrics": merged.anchoring_metrics}, ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        sys.exit(2)


if __name__ == "__main__":
    main()
