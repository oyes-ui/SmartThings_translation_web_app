#!/usr/bin/env python3
"""Compare agent/API proposals with a human-approved golden set.

The original v1 cell-level fields remain accepted.  V2 additionally exposes
severity and source coverage so four-week shadow runs can be judged safely.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load(value: str) -> Any:
    path = Path(value).expanduser()
    return json.loads(path.read_text(encoding="utf-8") if path.is_file() else value)


def evaluate(golden: list[dict], results: list[dict]) -> dict:
    expected = {str(item["id"]): item for item in golden}
    actual = {str(item["id"]): item for item in results}
    if set(expected) != set(actual):
        raise ValueError("golden/results id 집합이 일치하지 않습니다.")
    rows, source_counts = [], Counter()
    exact = expected_changes = false_positive = critical_missed = 0
    for item_id in sorted(expected):
        human, proposal = expected[item_id], actual[item_id]
        before = human.get("before")
        expected_after = human.get("expected_after", before)
        proposed_after = proposal.get("proposed_after", proposal.get("before"))
        expected_change, proposed_change = expected_after != before, proposed_after != proposal.get("before")
        match = proposed_after == expected_after
        severity = str(human.get("severity", "normal")).lower()
        missed = expected_change and not proposed_change
        critical_miss = missed and severity == "critical"
        exact += match
        expected_changes += expected_change
        false_positive += proposed_change and not expected_change
        critical_missed += critical_miss
        source = str(proposal.get("source", "agent"))
        source_counts[source] += 1
        rows.append({
            "id": item_id, "match": match, "expected_after": expected_after,
            "proposed_after": proposed_after, "severity": severity,
            "false_positive": proposed_change and not expected_change,
            "missed_change": missed, "critical_missed_change": critical_miss,
            "source": source,
        })
    total = len(expected)
    return {
        "total": total, "exact_matches": exact,
        "exact_match_rate": exact / total if total else 0,
        "expected_changes": expected_changes, "false_positive_changes": false_positive,
        "false_positive_rate": false_positive / total if total else 0,
        "critical_missed_changes": critical_missed,
        "shadow_exit_ready": critical_missed == 0,
        "source_coverage": dict(sorted(source_counts.items())),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("results")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        output = evaluate(load(args.golden), load(args.results))
        text = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
        print(text, end="")
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        sys.exit(2)


if __name__ == "__main__":
    main()
