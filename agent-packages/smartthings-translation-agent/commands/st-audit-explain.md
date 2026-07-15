# /st-audit-explain

검수 등급 또는 피드백이 왜 나왔는지 6대 항목 기준으로 설명한다.

## Arguments

`$ARGUMENTS`

권장 입력:

- 소스 문구
- 번역문
- 대상 언어
- 검수 등급 또는 피드백
- 필요하면 workbook 시트/셀

## Workflow

1. `references/response-patterns.md`의 "검수 등급 설명" 형식을 따른다.
2. `references/rules-sources.md`의 검수 6대 항목을 기준으로 분류한다.
3. RAG 사례가 필요하면 `/st-rag` 흐름을 제안하거나 함께 사용한다.
4. Needs Revision이면 수정안을 제시하되, Excel에는 적용하지 않는다.

## Rules

- 규칙 위반과 스타일 선호를 분리해서 말한다.
- 근거가 부족하면 불확실성을 명시한다.
- Excel 수정은 사용자가 별도 승인하기 전까지 하지 않는다.
