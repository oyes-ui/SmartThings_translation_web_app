# Claude for Excel Capability and Fallback Policy

## 왜 별도 문서인가

`excel-chatgpt/officejs-capabilities.md`는 ChatGPT for Excel이 사용자가 붙여넣은 Office.js
코드를 live session에서 실행해 준다는 전제(`officejs-probe.js` 참고)를 깐다. Claude for Excel이
같은 방식(임의 Office.js 코드 실행)을 제공하는지, 아니면 자체적으로 노출하는 고정된
read/write 도구(예: 활성 sheet 조회, range 읽기/쓰기 tool call) 형태인지는 세션마다 실제로
확인하기 전까지 가정하지 않는다. 아래 정책은 두 경우 모두에 적용되는 **capability-agnostic**
버전이다.

## 필수 확인

- 현재 연결된 Claude for Excel 세션이 다음을 제공하는지 실제로 확인한다: 활성 workbook/sheet
  조회, selection 또는 range 읽기(값·수식·text), sheet 보호·숨김·병합 상태 조회, range 쓰기.
- 어떤 형태로 노출되든(Office.js 코드 실행 경로, 고정 tool call, 또는 다른 방식) 위 항목을
  읽기 전용으로 먼저 검증한다. 이름이나 시그니처를 문서에 미리 하드코딩하지 않는다 —
  세션이 실제로 광고하는 기능만 신뢰한다.
- event handler(있다면) 는 session/add-in 재시작 뒤 재등록해야 한다. event만을 영구 감사
  기록으로 사용하지 않는다.

## Rich Text Policy

셀 내부 부분 문자열의 글자색은 SmartThings 납품 품질에 영향을 준다. Claude for Excel의
쓰기 기능이 해당 workbook에서 run 단위 서식(부분 문자열 색상)의 읽기·쓰기·보존을
지원한다는 검증이 없으면 live 적용을 금지한다.

- 허용: 매칭 후보 preview, 명시적으로 선택한 셀 단위 강조
- 보류: 검증되지 않은 부분 문자열 rich text 변경
- fallback: `workbook_highlight_glossary.py`를 통한 Delivery Python 복사본 생성 및 검증

Excel Add-in 플랫폼(Office.js) 공식 문서는 `Range.format.font` 기반의 **셀/range 단위**
글자색만 명시한다. Claude for Excel이 내부적으로 Office.js를 사용하더라도, substring run
단위 형식이 이 세션에서 읽기·쓰기·보존되는지는 별도로 실증해야 한다. 실증되기 전까지는
이를 납품 기능으로 간주하지 않는다.

## Glossary Source Policy

- 허용: 사용자가 import한 CSV → `__ST_GLOSSARY` table, 승인된 app API/MCP
- 금지: 임의 로컬 경로 탐색, `.env`/DB 직접 열람, 원본 경로 기록
- 기록: glossary locale, version, checksum, import timestamp
