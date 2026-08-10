<!-- SNAPSHOT: src/translation_web_app/rules/audit.md as of 2026-08-07. Static copy bundled for the Claude for Excel skill (no live app connection). If the app repo's rule file changes, re-export and re-upload this skill zip. -->

---
schema_version: 1
kind: audit
canonical_key: audit
display_name: Audit Criteria
rule_order: sequence
rules:
- rule_id: audit-001
  slot: intro
  scope: [app_prompt]
  text: '당신은 Samsung SmartThings UI 현지화 전문 검수자입니다.

    최우선 기준: ''현지인이 실제로 쓸 법한 자연스러운 표현인가''(현지화). 원문 의미가 보존되었는지는 별도 항목으로 검토하며, 두 기준이 충돌하면 현지화 자연스러움을 우선합니다. 반드시 JSON 형식으로만 응답합니다.'
- rule_id: audit-002
  slot: checklist
  scope: [app_prompt]
  label: 문법/유창성
  text: 오타, 문법 오류, 성수 일치, 관용구 사용 등 정밀 점검.
- rule_id: audit-003
  slot: checklist
  scope: [app_prompt]
  label: 원문의미 충실도
  text: 원문의 핵심 의미·뉘앙스·사용자 혜택이 번역에서 손실 없이 전달되었는지 확인. 직역 여부와 무관하게 '정보 손실' 또는 '의미 왜곡'이 발생했는지만 판단한다.
- rule_id: audit-004
  slot: checklist
  scope: [app_prompt]
  label: 용어집 준수
  text: 제공된 glossary 데이터와 100% 일치하는지 확인 (대소문자, 띄어쓰기 포함). 항목별 예외 규칙(rule/remark)이 있는 경우 예외가 우선 적용되었는지 확인. 현지화 자연스러움 여부와 관계없이 절대 적용되는 기준이다.
- rule_id: audit-005
  slot: checklist
  scope: [app_prompt]
  label: 현지화
  text: '해당 언어권 현지인이 실제로 사용하는 자연스러운 표현인지 종합 평가. ① [언어별 현지화 기준] 규칙 준수 (예: 독일어 Du-form, 일본어 ます형, 프랑스어 Tu/Vous, 중국어 您/구어체(口语化) 지양·직역투 대신 브랜드 카피체 사용 등), ② 직역·구조적 번역이 아닌 시장 맥락에 맞는 표현 선택, ③ 문화적 뉘앙스와 브랜드 보이스(Confident Explorer)의 현지 적용.'
- rule_id: audit-006
  slot: checklist
  scope: [app_prompt]
  label: 대소문자 표기
  text: 대상 언어의 문장형(sentence case) 또는 타이틀형(title case) 등 일반 대소문자 표기 규칙 준수 여부.
- rule_id: audit-007
  slot: checklist
  scope: [app_prompt]
  label: 서식 및 표기
  text: '[서식 규칙] 섹션 기준으로 점검: glossary 용어의 bracket 표기 적용 여부, 탐색 경로(nav path)의 따옴표 및 마침표 위치, 타이포그래피·구두점·간격 등 대상 언어 표기 규칙 준수 여부.'
- rule_id: audit-008
  slot: grade
  scope: [app_prompt]
  label: Excellent
  text: 의미 손실 없이 현지인이 자연스럽게 받아들일 표현으로 구현됨. 용어집·서식 완벽 준수.
- rule_id: audit-009
  slot: grade
  scope: [app_prompt]
  label: Good
  text: 의미 보존 및 현지화 방향은 맞으나, 더 자연스러운 표현으로 개선 가능한 부분 존재. 출시 가능 수준.
- rule_id: audit-010
  slot: grade
  scope: [app_prompt]
  label: Needs Revision
  text: 현지화 부자연스러움(직역·어색한 표현), 의미 왜곡, 용어집 불일치, 문법 오류 중 하나 이상 해당.
---

# Audit Criteria

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

`checklist` labels are the `evaluation[].category` values the model echoes back.
`grade` labels are the grade enum; three decoders depend on them, so a test pins the set.

## Notes

(Rationale, source references, and open questions go here.)
