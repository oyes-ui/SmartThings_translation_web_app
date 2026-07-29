---
name: smartthings-excel-live
description: ChatGPT for Excel에서 열린 SmartThings 번역 워크북을 읽고, 승인된 범위만 안전하게 수정하거나 glossary 하이라이트 후보를 검토한다. 현재 workbook·활성 sheet·선택 range를 대상으로 하는 요청에 사용한다.
---

# SmartThings Excel Live Layer

이 skill은 `smartthings-translation-agent`의 **live Excel 실행 레이어**다. 공통 번역 규칙,
RAG 우선순위, glossary 정책, report/manifest 계약은 상위 skill과 공유한다.

## 경계

- 현재 열려 있는 workbook, 활성 sheet, 사용자가 지정하거나 선택한 range만 다룬다.
- 로컬 Python, `openpyxl`, 임의의 로컬 파일 경로, `.env`, SQLite DB를 직접 실행하거나 읽지 않는다.
- 규칙/RAG/glossary는 workbook 내 `__ST_GLOSSARY` table 또는 승인된 SmartThings app API/MCP를 통해서만 받는다.
- Office.js capability가 없거나 rich text 보존이 검증되지 않은 경우 쓰기 작업을 하지 않고,
  Delivery Python 경로를 안내한다.

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
