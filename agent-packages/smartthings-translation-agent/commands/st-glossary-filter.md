# /st-glossary-filter

대상 언어의 source group 확정문구와 실제 SmartThings 용어집을 매칭해, 이번 파일에서 활성/비활성/수동 예외로 볼 용어를 판단한다.

## Arguments

`$ARGUMENTS`

권장 입력:

- workbook 경로
- 대상 언어 시트(예: BR/RU/CN)
- glossary CSV 경로(생략 시 app repo의 `runtime/glossary/latest_glossary.csv` 우선)
- 필요하면 source 시트명(자동 판별 우선)

## Workflow

1. `references/glossary-report-workflow.md`를 읽고 따른다.
2. 대상 언어의 source group을 판별한다: KR source는 `KR → JA/CN/TW/US`, US source는 `US → BR/RU/DE…`.
3. 선택된 source 시트를 `workbook_inspect.py`로 읽고 실제 glossary CSV의 해당 source 열과 매칭한다.
4. source 원문에서 `[ ... ]` bracket occurrence를 별도로 추출해 glossary와 대조한다.
5. 응답은 `references/response-patterns.md`의 "용어집 필터/활성화 판단" 형식을 사용한다.

## Rules

- source 문구만 보고 일반어를 추측하지 말고, 실제 glossary 매칭 결과만 제안한다.
- `비활성화` 용어라도 source 원문에 `[term]`으로 명시된 occurrence는 이번 파일/셀 한정 활성 후보로 본다.
- 같은 용어라도 bracket 없는 일반명사 occurrence는 수동 예외로 분리한다.
- Excel 원본과 glossary CSV는 사용자 승인 없이 수정하지 않는다.
