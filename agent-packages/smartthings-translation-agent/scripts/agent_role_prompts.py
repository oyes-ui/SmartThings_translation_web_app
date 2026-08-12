#!/usr/bin/env python3
"""Version-controlled prompts for the five /st-inspect specialist roles.

``build_role_prompt`` takes the shared evidence packet and nothing else.  That
signature is the enforcement for §4-A's independence rule: there is no parameter
through which another role's opinions could reach a specialist, so the anchoring
seen in the ES_CO batch (story_and_ui_coherence reusing semantic_fidelity's
``after`` values verbatim) cannot be reproduced through this path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_review_contract import SPECIALIST_ROLES  # noqa: E402

PROMPT_SCHEMA_VERSION = 1

# Each entry mirrors the §4-A role table: what the role decides, and the boundary
# that keeps the five perspectives from collapsing into one.
ROLE_BRIEFS: dict[str, dict[str, str]] = {
    "grammar_fluency": {
        "title": "문법·유창성",
        "judges": "대상 언어 문장이 문법적으로 맞고 자연스럽게 읽히는지. 관용구 직역, 어색한 연결·전치사·"
                  "격·어미, 마케팅 카피의 리듬을 본다.",
        "excludes": "원문과 의미가 같은지는 판단하지 않는다(semantic_fidelity의 몫). 문장이 원문 의미를 "
                    "바꿨다고 의심되면 수정안을 내지 말고 stance를 review로 남긴다.",
    },
    "semantic_fidelity": {
        "title": "원문 의미 충실도",
        "judges": "번역이 원문의 의미와 행위 주체를 보존하는지. 누가 누구에게 하는 동작인지, 능동/수동, "
                  "대상 관계가 뒤집히지 않았는지를 본다.",
        "excludes": "문장이 매끄러운지, 톤이 시장에 맞는지는 판단하지 않는다. 의미가 보존됐다면 어색함만을 "
                    "이유로 수정안을 내지 않는다.",
    },
    "localization_tone": {
        "title": "현지화·톤",
        "judges": "과거 번역 관행·시장 톤과 일치하는지. RAG 사례와 BX 스타일 기준이 주된 근거다.",
        "excludes": "문법 정오답과 원문 의미 보존 여부는 판단 대상이 아니다. RAG 사례는 규칙·glossary를 "
                    "이기지 못한다.",
    },
    "style_and_hard_rule_exceptions": {
        "title": "표기·스타일 / 하드룰 예외",
        "judges": "하드룰 바깥의 표기·스타일 일관성. 그리고 근거 패킷에 이미 계산된 하드룰 결과에 대해 "
                  "문맥상 예외로 볼 사유가 있는지만 판단한다.",
        "excludes": "glossary·대소문자·bracket·브랜드 하드룰을 다시 계산하거나 다른 결론으로 덮어쓰지 "
                    "않는다. 패킷의 값을 전제로만 판단한다.",
    },
    "story_and_ui_coherence": {
        "title": "story·section·UI 기능 조건",
        "judges": "title → description → CTA → disclaimer로 이어지는 지칭·기능명·사용자 혜택·조건의 연결, "
                  "그리고 source가 확인해 준 UI 활성화 조건이 유지되는지.",
        "excludes": "v1에는 대응하는 결정론적 구조 체커가 없다. UI navigation path는 실제 제품 UI 근거 "
                    "없이 추정 수정하지 않고 stance를 review로 남긴다.",
    },
}

# Evidence each role actually needs.  Sections, the cell snapshot and the resolved
# constraint card go to everyone; only the case-analysis internals are narrowed to
# the role that owns them.
#
# The card is not optional per role.  An earlier split withheld it from four of the
# five, leaving them a bare ``hard_rule_issues`` list — which is empty exactly when
# a term resolved cleanly, so "this cell has a settled glossary target" and "this
# cell has no constraints" arrived as the same signal.  Three roles then proposed
# their own wording over an already-correct glossary term (ES_CO C18, Movie mode).
ROLE_EVIDENCE_SLICES: dict[str, tuple[str, ...]] = {
    "grammar_fluency": ("sections", "deterministic_summary"),
    "semantic_fidelity": ("sections", "deterministic_summary"),
    "localization_tone": ("sections", "deterministic_summary", "candidate_overlay"),
    "style_and_hard_rule_exceptions": ("sections", "deterministic_full", "candidate_overlay"),
    "story_and_ui_coherence": ("sections", "deterministic_summary"),
}

# Fields of a deterministic_evidence entry each slice level exposes.  ``glossary_context``
# appears in neither: it is the merge-time gate's input, and a specialist handed it
# could re-derive the bracket/casing policy the card has already settled.
_SUMMARY_EVIDENCE_FIELDS = ("cell", "row_type", "hard_rule_issues",
                            "constraint_card", "constraint_validation")
_FULL_EVIDENCE_FIELDS = _SUMMARY_EVIDENCE_FIELDS + ("sentence_case_report", "simple_case_fix")

_CAPABILITY = """\
## 허용된 행동 (이 밖의 것은 하지 않는다)

- 허용: 아래 근거 패킷 읽기, offline RAG 조회, 승인된 예산 안에서의 semantic RAG 조회,
  JSON 의견서 1개 출력.
- 금지: Excel 파일 읽기·쓰기·적용, 납품/하이라이트 도구 실행, 워크북 경로 접근,
  다른 역할의 의견서 파일 열기, 저장소 파일 수정, 외부 네트워크·MCP 호출.
- 너는 읽기 전용 검수자다. 무엇도 적용하지 않으며 적용을 제안만 한다.
"""

_COMMON_RULES = """\
## 공통 규칙

1. 원본 Excel을 수정하지 않는다. 너는 의견서만 낸다.
2. 아래 근거 패킷 **밖의 정보를 새로 만들어내지 않는다**. 하드룰(glossary·대소문자·bracket·
   브랜드) 결과는 이미 계산되어 패킷에 들어 있으므로 재계산하지 않는다.
3. 다른 관점의 의견을 참고하지 않는다. 너에게는 제공되지 않으며, 추측해서 맞추려 하지 않는다.
4. 발견이 없으면 빈 `opinions`와 `status: "no_findings"`로 완료 의견서를 낸다. 침묵은 완료로
   인정되지 않는다.
5. 확신이 없으면 `stance: "review"`로 남긴다. 근거가 약한 수정안을 내는 것보다 낫다.
6. offline RAG 조회는 자유롭게 한다. semantic RAG는 아래 예산 안에서만 쓴다.
"""

_OUTPUT_CONTRACT = """\
## 출력 형식

아래 JSON 하나만 출력한다.

```json
{
  "schema_version": 1,
  "role": "<role>",
  "packet_id": "<packet_id>",
  "target_sheet": "<sheet>",
  "status": "completed | no_findings",
  "stop_reason": "complete",
  "started_at": "<ISO8601>",
  "finished_at": "<ISO8601>",
  "confidence": "high | medium | low",
  "rag_evidence_ids": [],
  "opinions": [
    {
      "role": "<role>",
      "sheet": "<sheet>",
      "cell": "C11",
      "finding_id": "story_012_C11_<짧은_설명>",
      "stance": "support | oppose | review",
      "reason": "왜 이렇게 판단했는지 (너의 관점 기준으로)",
      "after": "제안 번역문 (stance가 support일 때만; 아니면 null)",
      "rule_ids": ["근거 규칙 식별자"],
      "constraint_status": "pass | human_review | blocked"
    }
  ]
}
```

- `finding_id`는 같은 셀의 같은 쟁점이면 다른 관점과 자연히 겹치도록 `story_<번호>_<셀>_<쟁점>`
  형태로 짓는다.
- `after`는 셀 전체의 최종 문자열이다. 부분 조각이 아니다.
- 중간에 끊기거나 오류로 끝나면 `stop_reason`에 그 사유를 적고 `error`를 채운다. 정상 완료가
  아닌 의견서는 검수에서 제외되며, 그 편이 불완전한 결과가 조용히 반영되는 것보다 안전하다.
"""


def _evidence_slice(role: str, packet: dict[str, Any]) -> dict[str, Any]:
    """Give a role the evidence it needs and nothing heavier."""
    slices = ROLE_EVIDENCE_SLICES[role]
    evidence: dict[str, Any] = {key: packet.get(key) for key in (
        "workbook_name", "target_sheet", "source_sheet", "packet_id")}
    if "sections" in slices:
        evidence["target_sections"] = packet.get("target_sections")
        evidence["source_sections"] = packet.get("source_sections")
        evidence["cell_snapshot"] = packet.get("cell_snapshot")
    deterministic = packet.get("deterministic_evidence") or []
    fields = _FULL_EVIDENCE_FIELDS if "deterministic_full" in slices else _SUMMARY_EVIDENCE_FIELDS
    evidence["deterministic_evidence"] = [
        {key: item.get(key) for key in fields}
        for item in deterministic if isinstance(item, dict)
    ]
    if "candidate_overlay" in slices:
        evidence["candidate_overlay"] = packet.get("candidate_overlay")
    return evidence


def build_role_prompt(role: str, packet: dict[str, Any]) -> str:
    """Render the full instruction for one specialist from the shared packet."""
    if role not in SPECIALIST_ROLES:
        raise ValueError(f"알 수 없는 검수 관점: {role!r}")
    if not isinstance(packet, dict) or packet.get("kind") != "agent_sheet_review_packet":
        raise ValueError("packet은 agent_sheet_review.py가 만든 근거 패킷이어야 합니다.")
    if packet.get("review_mode") != "multi_agent":
        raise ValueError(
            "이 시트는 5개 관점 병렬 검수로 승인되지 않았습니다. "
            "기본 경로는 cell→sheet→lead이며, legacy escalation이 필요하면 "
            "agent_sheet_review.py를 --multi-agent로 다시 실행해 승인 패킷을 만드세요."
        )
    brief = ROLE_BRIEFS[role]
    budget = packet.get("semantic_rag_budget", 0)
    evidence = _evidence_slice(role, packet)
    preamble = packet.get("hard_constraint_preamble") or ""
    authority = f"\n## 하드 제약의 권위\n\n```\n{preamble.strip()}\n```\n" if preamble else ""
    return f"""\
# /st-inspect 전문 검수 — {brief['title']} (`{role}`)

너는 SmartThings 번역 검수의 **{brief['title']}** 관점 담당이다. 시트 `{packet.get('target_sheet')}`
전체를 이 관점에서만 독립적으로 검토한다.

## 네가 판단하는 것

{brief['judges']}

## 네가 판단하지 않는 것

{brief['excludes']}

{_CAPABILITY}
{_COMMON_RULES}
- semantic RAG 예산: **{budget}회** (0이면 offline만 사용한다)
{authority}
각 셀의 `constraint_card`는 앱 resolver가 이미 확정한 값이다. 여기 적힌 target 표기·활성화·
bracket 정책과 어긋나는 수정안은 병합 직전 같은 resolver에 다시 걸려 사람 검토로 넘어간다.
`hard_rule_issues`가 비어 있다는 것은 "제약이 없다"가 아니라 **"이미 통과했다"**는 뜻이다.

## 근거 패킷 (packet_id: `{packet.get('packet_id', '')}`)

```json
{json.dumps(evidence, ensure_ascii=False, indent=2)}
```

{_OUTPUT_CONTRACT.replace('<role>', role).replace('<sheet>', str(packet.get('target_sheet', ''))).replace('<packet_id>', str(packet.get('packet_id', '')))}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=list(SPECIALIST_ROLES))
    parser.add_argument("--packet", required=True)
    args = parser.parse_args()
    try:
        packet = json.loads(Path(args.packet).expanduser().read_text(encoding="utf-8"))
        print(build_role_prompt(args.role, packet))
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        sys.exit(2)


if __name__ == "__main__":
    main()
