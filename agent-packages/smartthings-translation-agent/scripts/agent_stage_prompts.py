#!/usr/bin/env python3
"""Build prompts for the default cell -> sheet consistency -> lead workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_staged_contract import validate_cell_review, validate_sheet_review


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def cell_prompt(packet: dict) -> str:
    evidence = [{key: item.get(key) for key in (
        "cell", "row_type", "source_text", "target_text", "hard_rule_issues",
        "constraint_card", "constraint_validation")}
        for item in packet.get("deterministic_evidence", [])]
    return f"""# /st-inspect 셀 순차 검수

한 명의 검수자로서 아래 셀을 배열 순서대로 각각 검수한다. 앱 audit처럼 각 셀의 원문·현재 번역·
constraint card를 함께 판단한다. resolver card는 최고 권위이며 덮어쓰지 않는다.

앞 셀을 참고했다면 숨기지 말고 `used_prior_cell_context`, `prior_cell_refs`,
`prior_cell_influence`에 정확히 기록한다. 참고하지 않았다면 false, [], 빈 문자열을 쓴다.
각 셀을 빠짐없이 한 번씩 출력하고 `after`는 수정이 필요한 경우 셀 전체 문자열이다.

출력은 JSON 하나다:
{{"kind":"cell_review","packet_id":"{packet.get('packet_id','')}","status":"completed",
"stop_reason":"complete","model":"<model>","cells":[
{{"cell":"C7","status":"pass|warning|needs_revision|blocked|glossary_activation_review",
"after":null,"reason":"...","rule_ids":[],"used_prior_cell_context":false,
"prior_cell_refs":[],"prior_cell_influence":""}}]}}

## 하드 제약
{packet.get('hard_constraint_preamble','')}

## 셀 근거
{_dump(evidence)}
"""


def sheet_prompt(packet: dict, cell_review: dict) -> str:
    validate_cell_review(cell_review, packet)
    return f"""# /st-inspect 시트 일관성 검수

전체 시트와 resolver 검증을 마친 셀 검수 결과를 보고 CTA·제품명·호칭·문체·UI 명칭의 통일성을
검토한다. **용어집에 없는 반복 표기**도 반드시 별도 후보화한다. 특히 같은 기능·기기를 가리키는
`TV / televisor`, `smartphone / celular`, `remote / control remoto`처럼 source/target 전반에 반복되는
명사·UI 라벨·CTA를 대조해 하나의 canonical lexical pattern을 정한다. 단수/복수·관사·문장 안의
정상적인 문법 변화는 오류가 아니며, 문체만 다른 경우에는 수정하지 않는다.

같은 referent와 비교 가능한 문맥에서 표기가 갈리면, 용어집 항목이 없어도 일관성 issue로 기록하고
affected_cells 각각에 셀 전체 수정안을 제시한다. 시트 안에 정답을 결정할 근거가 부족하면 임의로
통일하지 말고 issue에 `human_review_required`를 rule_ids에 넣고, 각 proposal의 `after`에는 현재
문자열을 그대로 넣는다. 이 no-op 후보는 변경안이 아니라 사람 검토 큐로 전환된다.
resolver_status가 pass가 아닌 제안을 정답으로 취급하지 않는다. 문제가 없으면 issues를 비운다.

출력 JSON:
{{"kind":"sheet_consistency_review","packet_id":"{packet.get('packet_id','')}",
"status":"completed|no_findings","stop_reason":"complete","model":"<model>","issues":[
{{"finding_id":"sheet_consistency_...","affected_cells":["C7"],"canonical_pattern":"...",
"reason":"...","rule_ids":[],"proposals":[{{"cell":"C7","after":"셀 전체 문자열","rule_ids":[]}}]}}]}}

## 시트 원문/번역
{_dump({'source_sections': packet.get('source_sections'), 'target_sections': packet.get('target_sections')})}

## 검증된 셀 결과
{_dump(cell_review)}
"""


def lead_prompt(packet: dict, cell_review: dict, sheet_review: dict) -> str:
    validate_cell_review(cell_review, packet)
    validate_sheet_review(sheet_review, packet)
    return f"""# /st-inspect 리드 통합

셀 검수와 시트 일관성 검수를 sheet+cell 기준으로 통합한다. 모든 셀을 배열 순서대로 판정한다.
새 쟁점이나 새 셀을 만들 수 없다. 수정문은 근거 범위 안에서 다듬을 수 있으며 basis_refs에
`cell:C7` 또는 해당 셀을 포함하는 `sheet:<finding_id>`를 기록한다. resolver blocked와
glossary_activation_review를 채택하지 않는다.

출력 JSON:
{{"kind":"lead_review","packet_id":"{packet.get('packet_id','')}","status":"completed",
"stop_reason":"complete","model":"<model>","decisions":[
{{"cell":"C7","finding_id":"lead-C7","status":"pass|warning|needs_revision|blocked|glossary_activation_review",
"after":null,"reason":"...","rule_ids":[],"basis_refs":["cell:C7"]}}]}}

## 셀 결과
{_dump(cell_review)}

## 시트 결과
{_dump(sheet_review)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("cell", "sheet", "lead"))
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--cell-review", type=Path)
    parser.add_argument("--sheet-review", type=Path)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    if packet.get("review_mode") != "staged_cell_sheet_lead":
        raise SystemExit("staged 기본 패킷이 아닙니다. --multi-agent 없이 패킷을 생성하세요.")
    cell = json.loads(args.cell_review.read_text(encoding="utf-8")) if args.cell_review else None
    sheet = json.loads(args.sheet_review.read_text(encoding="utf-8")) if args.sheet_review else None
    if args.stage == "cell":
        print(cell_prompt(packet))
    elif args.stage == "sheet":
        if cell is None or cell.get("resolver_gate_status") != "completed":
            raise SystemExit("resolver 검증을 마친 --cell-review가 필요합니다.")
        print(sheet_prompt(packet, cell))
    else:
        if cell is None or sheet is None or any(x.get("resolver_gate_status") != "completed" for x in (cell, sheet)):
            raise SystemExit("resolver 검증을 마친 cell/sheet review가 모두 필요합니다.")
        print(lead_prompt(packet, cell, sheet))


if __name__ == "__main__":
    main()
