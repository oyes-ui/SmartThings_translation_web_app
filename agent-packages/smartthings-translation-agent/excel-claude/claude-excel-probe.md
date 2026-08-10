# SmartThings Excel Live — read-only capability probe (Claude for Excel)

`excel-chatgpt/officejs-probe.js`는 ChatGPT for Excel이 사용자가 붙여넣은 Office.js 코드를
live session에서 실행해 준다는 것을 전제로 한 스크립트다. Claude for Excel이 같은 방식(임의
코드 실행)을 제공하는지, 아니면 자체 read/write 도구만 제공하는지는 세션마다 다를 수 있으므로
이 문서는 스크립트가 아니라 **확인해야 할 항목 체크리스트**로 둔다. Claude for Excel이 코드
실행 경로를 제공하는 것이 확인되면, 아래와 동일한 항목을 읽는 Office.js 스니펫(`excel-chatgpt/officejs-probe.js`
구조 재사용 가능)을 그 경로에서 실행해도 된다.

## 실행 전제

- Microsoft Excel에서 Claude for Excel add-in이 연결된 live session이어야 한다.
- 어떤 셀, workbook, table도 만들거나 수정하지 않는다. 이 probe는 읽기 전용이다.

## 확인 항목

| 항목 | 확인 방법 | 결과에 기록할 값 |
| --- | --- | --- |
| 활성 worksheet 이름·visibility | 세션이 제공하는 조회 기능으로 읽는다 | `worksheet.name`, `worksheet.visibility` |
| 현재 selection/range 주소 | 동일 | `selection.address` |
| 대상 셀의 value/formula/text | 동일 | `cell.firstValue`, `cell.firstFormula`, `cell.firstText` |
| sheet 보호 상태 | 동일 | `protected: true/false` |
| 병합 셀 여부 | **추측하지 않는다** — value가 없고 text만 있으면 `unknown: inspect manually before writing`로 표시 | `mergedState` |
| range 쓰기 기능 존재 여부 | 세션이 명시적으로 광고하는지 확인 | `writeRange: "requires separately advertised live command and approval"` |
| 부분 문자열(run 단위) rich text 지원 | 공식적으로 검증된 바 없다고 가정 | `partialRichText: "not asserted"` |
| glossary table(`__ST_GLOSSARY`) 존재 여부 | 별도 CSV import + 승인된 쓰기 단계 필요 | `glossaryTable: "requires explicit CSV import plus a separately approved write step"` |

## 결과 판정

- `writePerformed: false`
- `decision: preview_only`
- `fallback: "Use Delivery Python for glossary substring rich-text highlighting."`

위 항목 중 하나라도 세션에서 실제로 확인하지 못했다면 preview 단계를 넘어가지 않는다.
