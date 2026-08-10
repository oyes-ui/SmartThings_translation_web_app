---
name: smartthings-excel-live
description: ChatGPT for Excel에서 열린 SmartThings 번역 워크북을 읽고, 승인된 범위만 안전하게 수정하거나 glossary 하이라이트 후보를 검토한다. 현재 workbook·활성 sheet·선택 range를 대상으로 하는 요청에 사용한다.
---

# SmartThings Excel Live Layer

이 skill은 `smartthings-translation-agent`의 **live Excel 실행 레이어**다. 공통 번역 규칙,
RAG 우선순위, glossary 정책, report/manifest 계약은 상위 skill과 공유하며, 같은 목적의
[`excel-claude/`](../excel-claude/SKILL.md) 레이어(Claude for Excel용)와 안전 원칙·manifest
구조를 그대로 공유한다. 다른 점은 대상 surface(ChatGPT for Excel)와 그 도구 노출 방식뿐이다.

## 경계

- 현재 열려 있는 workbook, 활성 sheet, 사용자가 지정하거나 선택한 range만 다룬다.
- 로컬 Python, `openpyxl`, 임의의 로컬 파일 경로, `.env`, SQLite DB를 직접 실행하거나 읽지 않는다.
- 규칙/RAG/glossary는 workbook 내 `__ST_GLOSSARY` table 또는 승인된 SmartThings app API/MCP를 통해서만 받는다. 단, workbook glossary가 없으면 이 레이어에 번들된 정적 `references/` snapshot을 참고한다.
- Office.js capability가 없거나 rich text 보존이 검증되지 않은 경우 쓰기 작업을 하지 않고,
  Delivery Python 경로를 안내한다.

## 규칙 우선순위와 전체 참조

다음 우선순위를 따른다: (1) 사용자 승인·작업 범위·campaign brief, (2) workbook의 승인된
`__ST_GLOSSARY` 및 target locale, (3) `references/canonical-rules/glossary.md`, (4) target locale의
`references/canonical-rules/languages/<language>.md`, (5) `common.md`·`typography.md`·`bx_style.md`·`audit.md`,
(6) `references/workbook-role-and-casing.md`, (7) es_CO/glossary 정적 snapshot이다.

glossary의 **철자, 대소문자, 공백, 시장별 표기**는 heading/button의 title·sentence case보다 항상 우선한다.
검수와 preview에서는 문법/자연스러움, 의미 충실도, glossary, 현지화, 대소문자, 포맷·BX 6항목을 독립적으로 확인한다.

## Title / heading / casing 점검

- 모든 대상은 title, section heading, body, button/CTA, UI path, legal/disclaimer 중 역할을 먼저 판별한다. SmartThings Story Contents의 title 역할 셀은 보통 `C7`, `C10`, `C15`, `C20`, `C25`지만, 주소만으로 단정하지 않고 문맥을 확인한다.
- title/heading은 끝 마침표 `.`/`。`를 쓰지 않는다. 의문형 title의 `?`/`？`, URL, 버전, 약어, source가 요구하는 생략부호는 유지한다. body나 legal 문장에 이 규칙을 적용하지 않는다.
- **Title Case는 영어식 단어별 대문자화 지시가 아니라 제목 역할을 뜻한다.** target 언어의 sentence case·자연스러운 대소문자를 따른다. 특히 Swedish, Vietnamese, French, Russian에는 영어식 title case를 강제하지 않는다.
- es_CO는 기본 **tú**, vos/voseo 금지, neutral Latin American vocabulary, ustedes 복수형을 쓴다. tú 활용·명령형 일관성은 검수에서 별도로 확인한다.
- case/문장부호 수정 preview에는 `content_role`, `current`, `proposed`, `applied rule`, `glossary_override`를 기록한다. 역할·locale·glossary 적용이 불명확하면 `needs_review`로 남기고 자동 수정하지 않는다.

## 편집 절차

1. 대상 workbook·sheet·range와 현재값/수식/보호 상태를 읽는다.
2. `before → after`, 영향 범위, 수식·병합·rich text 위험을 preview로 보여준다.
3. 사용자의 명시적 승인을 받은 뒤에만 승인된 range를 수정한다.
4. 수정 후 대상 range를 다시 읽어 실제 값, 수식 오류, 허용 범위 밖 변경 여부를 확인한다.
5. 변경 요약과 glossary version/checksum을 manifest 또는 리포트에 남긴다. 원본 경로와 secret은 기록하지 않는다.

## Glossary 하이라이팅

`/st-edit --highlight-glossary`의 하위 workflow로 사용한다.

- 기본은 매칭 preview다: glossary term, 대상 sheet/cell, 일치 문자열, glossary version을 보여준다.
- 사용자가 CSV를 명시적으로 가져오면 숨김/보호 `__ST_GLOSSARY` table에 저장할 수 있다.
- 부분 문자열 rich text 하이라이트는 Office.js capability와 보존 검증을 통과한 경우에만 적용한다.
- 검증되지 않은 경우 셀 전체 강조로 조용히 대체하지 않는다. 사용자가 셀 단위 강조를 명시하지 않으면 Delivery Python 경로로 보낸다.

## Delivery Python 경로

live preview는 `approval: "pending"` manifest만 만들며, 사람 승인 뒤 `approval: "approved"`와
각 변경의 `verification: "verified"`를 기록한다. `scripts/excel_live_manifest.py --require-approved`가
이를 검증하고, `workbook_apply_edits.py`(draft) 또는 `workbook_story_apply.py`(delivery)가 같은
manifest를 소비한다. `preview`/`blocked`/`fallback_delivery`는 적용 불가다.

납품본의 용어 단위 rich text 하이라이트와 전체 workbook 검증은 상위 skill의
`scripts/workbook_highlight_glossary.py` 및 관련 검증 workflow가 담당한다. 이 skill은
그 결과를 열린 workbook에서 확인하는 데만 사용한다.
