#!/usr/bin/env python3
"""Apply the app resolver to cell or sheet-stage agent output."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agent_sheet_merge import resolver_validator
from agent_staged_contract import gate_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("cell_review", "sheet_consistency_review"))
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--glossary", required=True, type=Path)
    parser.add_argument("--app-root", required=True, type=Path)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    validate = resolver_validator(packet, args.glossary.expanduser(), args.app_root.expanduser(),
                                  str(packet.get("target_sheet", "")))
    result = gate_stage(payload, args.stage, packet, validate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": "ok", "output": str(args.output),
                      "resolver_gate": result["resolver_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
