#!/usr/bin/env python3
"""Merge specialist opinions into a v2 review report and pending-approval manifest.

This is the general replacement for the one-off es_co_merge_st_inspect.py: it
takes any sheet's evidence packet plus one JSON file per specialist role, applies
the consensus gate, and emits artifacts that /st-apply can actually consume.
The original workbook is never modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_review_contract import (  # noqa: E402
    RUN_METADATA_FIELDS, SPECIALIST_ROLES, merge_subjective_opinions,
)
from review_report_builder import build_review_artifacts, write_artifacts  # noqa: E402


def load_role_opinions(opinions_dir: Path, packet_id: str) -> tuple[list[dict[str, Any]], list[str], dict[str, dict]]:
    """Read ``{role}.json`` per specialist, with its execution metadata.

    A missing file is not an error here — merge_subjective_opinions turns it into
    the ``incomplete`` sheet status, which is what §4-A asks for.  A file that
    reports an error or a truncated stop_reason is treated the same way.
    """
    opinions: list[dict[str, Any]] = []
    found: list[str] = []
    runs: dict[str, dict[str, Any]] = {}
    for role in SPECIALIST_ROLES:
        path = opinions_dir / f"{role}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("role") != role:
            raise ValueError(f"역할 불일치: {path} (role={payload.get('role')!r})")
        found.append(role)
        runs[role] = {key: payload.get(key) for key in RUN_METADATA_FIELDS if key in payload}
        for opinion in payload.get("opinions", []):
            # The packet id may live on the payload rather than each opinion.
            opinions.append({**opinion, "packet_id": opinion.get("packet_id") or payload.get("packet_id") or packet_id})
    return opinions, found, runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, help="agent_sheet_review.py가 만든 근거 패킷 JSON")
    parser.add_argument("--opinions-dir", required=True, help="역할별 {role}.json이 있는 디렉터리")
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--source-file-id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deterministic-proposals", help="하드룰 위반 제안 JSON list (합의 게이트 면제)")
    parser.add_argument("--trust-opinion-packet-id", action="store_true",
                        help="의견서에 기록된 packet_id를 그대로 검증한다(기본은 누락 시 패킷 값으로 채움)")
    args = parser.parse_args()
    try:
        packet = json.loads(Path(args.packet).expanduser().read_text(encoding="utf-8"))
        if packet.get("review_mode") == "lead_2pass":
            raise ValueError(
                "이 시트는 5개 관점 병렬 검수로 승인되지 않았습니다(review_mode=lead_2pass). "
                "agent_sheet_review.py를 --multi-agent로 실행해 승인 패킷을 먼저 만드세요."
            )
        packet_id = str(packet.get("packet_id", ""))
        opinions, found, role_runs = load_role_opinions(
            Path(args.opinions_dir).expanduser(), "" if args.trust_opinion_packet_id else packet_id
        )
        deterministic = []
        if args.deterministic_proposals:
            deterministic = json.loads(Path(args.deterministic_proposals).expanduser().read_text(encoding="utf-8"))
            if not isinstance(deterministic, list):
                raise ValueError("deterministic-proposals는 list여야 합니다.")
        merged = merge_subjective_opinions(opinions, packet=packet, deterministic_proposals=deterministic,
                                           role_runs=role_runs)
        manifest, markdown = build_review_artifacts(
            Path(args.workbook).expanduser(), merged,
            report_id=args.report_id,
            source_file_id=args.source_file_id or packet.get("workbook_name", ""),
        )
        paths = write_artifacts(manifest, markdown, args.output_dir, args.report_id)
        print(json.dumps({
            "status": "ok", **paths,
            "sheet_status": merged.sheet_status,
            "roles_found": found,
            "missing_roles": merged.missing_roles,
            "changes": len(manifest["changes"]),
            "human_review_queue": len(manifest["review_context"]["human_review_queue"]),
            "anchoring_warnings": len(merged.anchoring),
        }, ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        sys.exit(2)


if __name__ == "__main__":
    main()
