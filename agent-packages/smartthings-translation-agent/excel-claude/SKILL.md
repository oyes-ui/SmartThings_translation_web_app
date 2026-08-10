---
name: smartthings-excel-live-msoffice
description: Claude for Excel에서 열린 SmartThings 번역 워크북을 읽고, 승인된 범위만 안전하게 수정하거나 glossary 하이라이트 후보를 검토한다. 현재 workbook·활성 sheet·선택 range를 대상으로 하는 요청에 사용한다.
---

# SmartThings Excel Live Layer (Claude for Excel)

이 skill은 `smartthings-translation-agent`의 **live Excel 실행 레이어**다. 공통 번역 규칙,
RAG 우선순위, glossary 정책, report/manifest 계약은 상위 skill과 공유하며, 같은 목적의
[`excel-chatgpt/`](../excel-chatgpt/SKILL.md) 레이어(ChatGPT for Excel용)와 안전 원칙·manifest
구조를 그대로 공유한다. 다른 점은 대상 surface(Claude for Excel)와 그 도구 노출 방식뿐이다.

## 경계

- 현재 열려 있는 workbook, 활성 sheet, 사용자가 지정하거나 선택한 range만 다룬다.
- 로컬 Python, `openpyxl`, 임의의 로컬 파일 경로, `.env`, SQLite DB를 직접 실행하거나 읽지 않는다.
- 규칙/glossary는 `references/`에 번들된 정적 스냅샷, workbook 내 `__ST_GLOSSARY` table,
  또는 승인된 SmartThings app API/MCP에서만 받는다. 이 skill 자체에는 RAG(과거 번역 사례)
  조회 기능이 없다 — RAG가 필요하면 Delivery Python 경로(`scripts/rag_lookup.py`)로 안내한다.
- Claude for Excel이 제공하는 workbook read/write 기능(정확한 이름과 범위는 세션마다 확인)이
  rich text 보존을 검증하지 못한 경우 쓰기 작업을 하지 않고, Delivery Python 경로를 안내한다.

## 판단 기준 (references/)

이 skill은 "안전하게 읽고 쓰는 절차"뿐 아니라 **es_CO 번역 판단 기준**도 번들로 갖고 있다.
전부 app repo(`src/translation_web_app/rules/`, `runtime/glossary/`)의 **2026-08-07 시점 정적
스냅샷**이며, 각 파일 상단에 원본 경로와 스냅샷 날짜가 주석으로 남아 있다. app 쪽 규칙이
바뀌면 이 스킬은 자동으로 반영되지 않으므로, 중요한 작업 전에는 최신 여부를 사용자에게
확인한다.

| 파일 | 내용 | app repo 원본 |
| --- | --- | --- |
| `references/rules-es-CO.md` | tú 기본, vos 금지, 라틴아메리카 어휘 등 CO 전용 규칙 | `rules/languages/Spanish_Colombia.md` |
| `references/rules-common.md` | 전 언어 공통 품질 기준(의도 보존, 직역 회피 등) | `rules/common.md` |
| `references/rules-glossary-format.md` | 용어집 표기·대괄호·nav path·disclaimer 서식 규칙 | `rules/glossary.md` |
| `references/rules-audit-checklist.md` | 검수 6대 항목과 Excellent/Good/Needs Revision 등급 기준 | `rules/audit.md` |
| `references/rules-typography.md` | 구두점·간격·케이스 | `rules/typography.md` |
| `references/rules-bx-style.md` | Samsung BX 브랜드 보이스(Confident Explorer) | `rules/bx_style.md` |
| `references/glossary-co-slim.md` | term/rule/ko_KR/en_US/es_ES/es_CO 6열만 남긴 CO 중심 glossary 스냅샷(145항목, 2026-08-03 기준) | `runtime/glossary/latest_glossary_260803.csv` |
| `references/glossary-activation-guide.md` | glossary `rule` 값(빈값/`대괄호 제외`/`비활성화`)을 occurrence 단위로 판단하는 절차 — **term 단위로 일괄 활성/비활성 처리하지 않는다** | `references/glossary-report-workflow.md`에서 발췌·각색 |

`glossary-co-slim.md`을 쓸 때는 반드시 `glossary-activation-guide.md`의 판단 절차를 같이
적용한다. `rule`이 `비활성화`라고 곧바로 무시하거나, 빈 값이라고 곧바로 강제 적용하면 안 된다
— 두 경우 모두 source 원문의 대괄호 occurrence를 먼저 확인해야 한다. 워크북에 `__ST_GLOSSARY`
table이 별도로 import돼 있으면 그쪽이 이 정적 스냅샷보다 우선한다.

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
- 부분 문자열 rich text 하이라이트는 Claude for Excel의 쓰기 기능이 run 단위 서식 보존을
  지원한다고 검증된 경우에만 적용한다.
- 검증되지 않은 경우 셀 전체 강조로 조용히 대체하지 않는다. 사용자가 셀 단위 강조를 명시하지 않으면 Delivery Python 경로로 보낸다.

## Delivery Python 경로

live preview는 `approval: "pending"` manifest만 만들며, 사람 승인 뒤 `approval: "approved"`와
각 변경의 `verification: "verified"`를 기록한다. `scripts/excel_live_manifest.py --require-approved`가
이를 검증하고, `workbook_apply_edits.py`(draft) 또는 `workbook_story_apply.py`(delivery)가 같은
manifest를 소비한다. `preview`/`blocked`/`fallback_delivery`는 적용 불가다.

납품본의 용어 단위 rich text 하이라이트와 전체 workbook 검증은 상위 skill의
`scripts/workbook_highlight_glossary.py` 및 관련 검증 workflow가 담당한다. 이 skill은
그 결과를 열린 workbook에서 확인하는 데만 사용한다.
