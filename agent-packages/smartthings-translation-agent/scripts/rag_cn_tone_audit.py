#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_cn_tone_audit.py — 간체 중국어 RAG 과거 사례 구어체 톤 오디트 (크레딧 0)

RAG DB의 CN(중국) 과거 번역 사례를 스캔해, 어기조사(呀/啦/呢/哦/嘛) 등 구어체
신호가 있는 row를 리포트로 뽑는다. 삭제/수정은 하지 않는다 — 사람이 리포트를
검토해 제외 대상을 확정한 뒤, 별도 단계(tone_flag 마이그레이션)에서 처리한다.

사용 예:
  python rag_cn_tone_audit.py                       # notes/ 에 마크다운 리포트 저장
  python rag_cn_tone_audit.py --json                 # stdout에 JSON 출력
  python rag_cn_tone_audit.py --out /path/report.md  # 저장 경로 지정
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_lookup import _connect_sqlite  # noqa: E402  (import 시 bootstrap 부수효과 포함)

TARGET_LANG = "CN(중국)"

HIGH_CONFIDENCE_PATTERN = re.compile(r"[呀啦呢哦嘛]")
SOFT_SIGNAL_KEYWORDS = ["别让", "甚至无需", "就能", "轻松", "无需担心"]
INFORMAL_PRONOUN = "你"
FORMAL_PRONOUN = "您"


def scan_row(target_text: str) -> dict:
    text = target_text or ""
    return {
        "high_confidence": sorted(set(HIGH_CONFIDENCE_PATTERN.findall(text))),
        "soft_signals": [kw for kw in SOFT_SIGNAL_KEYWORDS if kw in text],
        "uses_ni": INFORMAL_PRONOUN in text,
        "uses_nin": FORMAL_PRONOUN in text,
    }


def run_audit(conn) -> dict:
    rows = conn.execute(
        "SELECT id, story_id, section_code, source_text, target_text "
        "FROM rag_pairs WHERE target_lang = ? ORDER BY story_id, section_code",
        (TARGET_LANG,),
    ).fetchall()

    high_confidence_rows = []
    soft_only_rows = []
    ni_count = 0
    nin_count = 0

    for row_id, story_id, section_code, source_text, target_text in rows:
        signals = scan_row(target_text)
        if signals["uses_ni"]:
            ni_count += 1
        if signals["uses_nin"]:
            nin_count += 1

        entry = {
            "id": row_id,
            "story_id": story_id,
            "section_code": section_code,
            "source_text": source_text,
            "target_text": target_text,
            **signals,
        }
        if signals["high_confidence"]:
            high_confidence_rows.append(entry)
        elif signals["soft_signals"]:
            soft_only_rows.append(entry)

    return {
        "total": len(rows),
        "high_confidence_rows": high_confidence_rows,
        "soft_only_rows": soft_only_rows,
        "ni_count": ni_count,
        "nin_count": nin_count,
    }


def _md_escape(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def _render_table(rows: list[dict], signal_key: str, signal_label: str) -> list[str]:
    if not rows:
        return ["(없음)"]
    lines = [f"| id | story_id | section_code | source_text | target_text | {signal_label} |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        signals = r[signal_key]
        signal_str = "".join(signals) if signal_key == "high_confidence" else ", ".join(signals)
        lines.append(
            f"| {r['id']} | {r['story_id']} | {r['section_code']} | "
            f"{_md_escape(r['source_text'])} | {_md_escape(r['target_text'])} | {signal_str} |"
        )
    return lines


def render_markdown(result: dict) -> str:
    lines = [f"# 간체 중국어 RAG 톤 오디트 ({datetime.now():%Y-%m-%d})", ""]
    lines.append(f"- 전체 CN(중국) row: {result['total']}건")
    lines.append(f"- High-confidence(어기조사 呀/啦/呢/哦/嘛) 플래그: {len(result['high_confidence_rows'])}건")
    lines.append(f"- Soft signal만 매칭(어기조사 없음, 참고용): {len(result['soft_only_rows'])}건")
    lines.append(f"- `你` 사용: {result['ni_count']}건 / `您` 사용: {result['nin_count']}건 "
                 f"(참고 카운트 — 단독으로는 플래그하지 않음)")
    lines.append("")
    lines.append("## High-confidence (구어체 어기조사 포함)")
    lines.append("")
    lines.extend(_render_table(result["high_confidence_rows"], "high_confidence", "매칭 어기조사"))
    lines.append("")
    lines.append("## Soft signal만 매칭 (참고용 — 삭제/제외 판단에 단독 사용 금지)")
    lines.append("")
    lines.extend(_render_table(result["soft_only_rows"], "soft_signals", "soft signals"))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="간체 중국어 RAG 과거 사례 구어체 톤 오디트 (크레딧 0)")
    parser.add_argument("--out", help="리포트 저장 경로 (기본: notes/cn-tone-audit-<날짜>.md)")
    parser.add_argument("--json", action="store_true", help="stdout에 JSON 출력 (파일 저장 없음)")
    args = parser.parse_args()

    conn = _connect_sqlite()
    try:
        result = run_audit(conn)
    finally:
        conn.close()

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    report = render_markdown(result)
    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "notes" / f"cn-tone-audit-{datetime.now():%Y-%m-%d}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(report, encoding="utf-8")
    os.replace(tmp, out_path)

    print(f"✅ 리포트 저장: {out_path}")
    print(f"   전체: {result['total']}건 / high-confidence: {len(result['high_confidence_rows'])}건 / "
          f"soft-only: {len(result['soft_only_rows'])}건")


if __name__ == "__main__":
    main()
