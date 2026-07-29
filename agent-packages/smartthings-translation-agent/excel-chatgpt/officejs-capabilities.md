# Office.js Capability and Fallback Policy

## 필수 확인

- Excel requirement set과 현재 session이 필요한 live workbook read/write 기능을 지원하는지 확인한다.
- workbook의 sheet 보호, 병합 범위, 수식 오류와 공동 편집에 따른 최신 상태를 읽는다.
- event handler는 session/add-in 재시작 뒤 재등록해야 한다. event만을 영구 감사 기록으로 사용하지 않는다.

## Rich Text Policy

셀 내부 부분 문자열의 글자색은 SmartThings 납품 품질에 영향을 준다. Office.js가 해당
workbook에서 run 단위 서식의 읽기·쓰기·보존을 지원한다는 검증이 없으면 live 적용을 금지한다.

- 허용: 매칭 후보 preview, 명시적으로 선택한 셀 단위 강조
- 보류: 검증되지 않은 부분 문자열 rich text 변경
- fallback: `workbook_highlight_glossary.py`를 통한 Delivery Python 복사본 생성 및 검증

공식 Excel JavaScript API 문서는 `Range.format.font` 기반의 **셀/range 단위** 글자색만 명시한다.
따라서 substring run 단위 형식이 같은 live session에서 읽기·쓰기·보존되는 실증이 있기 전까지는
이를 납품 기능으로 간주하지 않는다. read-only 실행은 `officejs-probe.js`, 실제 절차는
`live-runbook.md`를 따른다.

## Glossary Source Policy

- 허용: 사용자가 import한 CSV → `__ST_GLOSSARY` table, 승인된 app API/MCP
- 금지: 임의 로컬 경로 탐색, `.env`/DB 직접 열람, 원본 경로 기록
- 기록: glossary locale, version, checksum, import timestamp
