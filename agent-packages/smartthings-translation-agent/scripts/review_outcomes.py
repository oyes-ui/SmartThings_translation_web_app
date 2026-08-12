#!/usr/bin/env python3
"""Capture what a human decided about each proposal, and score against it.

Every review cycle a person already judges each proposed change.  Until now that
judgement was spent and discarded: the manifest was edited, approved rows went to
Excel, and rejected rows simply vanished.  Nothing recorded that the agent had
proposed something a human turned down, so accuracy could never be measured and
the approval gate could never be safely relaxed.

This module turns that existing decision into a durable ledger, and derives the
golden/results pair that quality_scorecard.py already knows how to evaluate.  It
never applies anything and never decides anything on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APPROVED = "approved"
DECISIONS = ("approved", "edited", "rejected")


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _suspect_roles(manifest: dict) -> set[str]:
    """Roles the anchoring check flagged as echoing another role."""
    suspect = set()
    for flag in manifest.get("anchoring") or []:
        if flag.get("role"):
            suspect.add(flag["role"])
        if flag.get("direction") == "undetermined" and flag.get("echoes_role"):
            suspect.add(flag["echoes_role"])
    return suspect


def outcomes_from_manifest(manifest: dict) -> list[dict[str, Any]]:
    """Read a human-reviewed manifest into ledger entries.

    ``proposed_after`` is the agent's frozen proposal and ``after`` is whatever
    the reviewer left behind, so an edit is distinguishable from a plain approval.
    """
    if not isinstance(manifest, dict) or not isinstance(manifest.get("changes"), list):
        raise ValueError("manifest는 changes 배열을 가진 객체여야 합니다.")
    suspect = _suspect_roles(manifest)
    recorded_at = datetime.now(timezone.utc).isoformat()
    entries = []
    for index, change in enumerate(manifest["changes"]):
        if not isinstance(change, dict):
            raise ValueError(f"changes[{index}]는 객체여야 합니다.")
        status = _text(change.get("approval_status")).strip().lower()
        if status == "pending_approval":
            raise ValueError(
                f"changes[{index}] ({change.get('finding_id')})가 아직 pending_approval입니다. "
                "사람이 승인/거절을 마친 manifest만 기록할 수 있습니다."
            )
        proposed = _text(change.get("proposed_after", change.get("after")))
        final = _text(change.get("after"))
        before = _text(change.get("before"))
        if status == APPROVED:
            decision = "approved" if final == proposed else "edited"
        else:
            decision = "rejected"
            final = before  # a rejected proposal leaves the cell as it was
        roles = list(change.get("supporting_roles") or [])
        entries.append({
            "finding_id": change.get("finding_id", ""),
            "report_id": manifest.get("report_id", ""),
            "packet_id": manifest.get("packet_id", ""),
            "sheet": change.get("sheet", ""),
            "cell": change.get("cell", ""),
            "row_type": change.get("row_type", ""),
            "origin": change.get("origin", "subjective_consensus"),
            "supporting_roles": roles,
            # Evidence that leaned on an echoing role is not independent, so it is
            # excluded from the scorecard rather than inflating the pass rate.
            "independence": "suspect" if (suspect and suspect.intersection(roles)) else "verified",
            "before": before,
            "proposed_after": proposed,
            "final_after": final,
            "decision": decision,
            "rejection_reason": _text(change.get("rejection_reason")),
            "severity": _text(change.get("severity") or "normal").lower(),
            "recorded_at": recorded_at,
        })
    return entries


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_ledger(path: Path, entries: list[dict[str, Any]]) -> int:
    """Append entries, replacing any earlier record of the same finding."""
    existing = load_ledger(path)
    incoming = {(entry["report_id"], entry["finding_id"]) for entry in entries}
    kept = [entry for entry in existing if (entry.get("report_id"), entry.get("finding_id")) not in incoming]
    merged = kept + entries
    atomic_write(path, "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in merged))
    return len(merged)


def scorecard_inputs(ledger: list[dict[str, Any]], *, verified_only: bool = True) -> tuple[list[dict], list[dict]]:
    """Build the (golden, results) pair quality_scorecard.evaluate expects."""
    rows = [row for row in ledger if not verified_only or row.get("independence") == "verified"]
    golden = [{"id": row["finding_id"], "before": row["before"],
               "expected_after": row["final_after"], "severity": row.get("severity", "normal")}
              for row in rows]
    results = [{"id": row["finding_id"], "before": row["before"],
                "proposed_after": row["proposed_after"], "source": row.get("origin", "agent")}
               for row in rows]
    return golden, results


def summarise(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-tier decision counts — the evidence for relaxing the approval gate.

    Reported per ``origin`` because that is the risk tier: deterministic hard-rule
    findings are the first candidates for autonomy, subjective consensus is not.
    """
    tiers: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(DECISIONS, 0))
    row_types: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(DECISIONS, 0))
    suspect = 0
    for row in ledger:
        decision = row.get("decision")
        if decision not in DECISIONS:
            continue
        tiers[row.get("origin", "unknown")][decision] += 1
        row_types[row.get("row_type") or "(미분류)"][decision] += 1
        suspect += row.get("independence") == "suspect"

    def rate(counts: dict[str, int]) -> dict[str, Any]:
        total = sum(counts.values())
        return {**counts, "total": total,
                "accepted_as_proposed_rate": round(counts["approved"] / total, 3) if total else 0.0}

    return {
        "entries": len(ledger),
        "independence_suspect": suspect,
        "by_origin": {key: rate(value) for key, value in sorted(tiers.items())},
        "by_row_type": {key: rate(value) for key, value in sorted(row_types.items())},
        "note": "이 요약은 측정값일 뿐이며 어떤 항목도 자동 적용하지 않는다.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="누적 기록 JSONL 경로")
    parser.add_argument("--manifest", help="사람이 승인/거절을 마친 manifest; 주면 ledger에 기록한다")
    parser.add_argument("--emit-golden", help="quality_scorecard용 golden set 출력 경로")
    parser.add_argument("--emit-results", help="quality_scorecard용 results 출력 경로")
    parser.add_argument("--include-suspect", action="store_true",
                        help="독립성이 의심되는 근거도 scorecard 입력에 포함한다(기본 제외)")
    args = parser.parse_args()
    try:
        ledger_path = Path(args.ledger).expanduser()
        recorded = 0
        if args.manifest:
            manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
            entries = outcomes_from_manifest(manifest)
            append_ledger(ledger_path, entries)
            recorded = len(entries)
        ledger = load_ledger(ledger_path)
        if args.emit_golden or args.emit_results:
            golden, results = scorecard_inputs(ledger, verified_only=not args.include_suspect)
            if args.emit_golden:
                atomic_write(Path(args.emit_golden).expanduser(),
                             json.dumps(golden, ensure_ascii=False, indent=2) + "\n")
            if args.emit_results:
                atomic_write(Path(args.emit_results).expanduser(),
                             json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": "ok", "recorded": recorded, "ledger": str(ledger_path),
                          "summary": summarise(ledger)}, ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        sys.exit(2)


if __name__ == "__main__":
    main()
