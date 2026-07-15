# /st-audit-explain

검수 등급 또는 피드백을 6대 항목 기준으로 재평가해 설명한다. AI 등급은 검토 후보이며 최종 판단이 아니다.

## Arguments

`$ARGUMENTS`

권장 입력:

- 소스 문구
- 번역문
- 대상 언어
- 검수 등급 또는 피드백
- 필요하면 workbook 시트/셀

## Workflow

1. 실제 workbook 셀과 AI 코멘트/수정안을 함께 확인한다. `Good`이라도 코멘트나 수정안이 있으면 재검토한다.
2. `references/response-patterns.md`의 "검수 등급 설명" 형식을 따라 `AI 판정`과 `최종 판단`을 분리해 출력한다.
3. `references/rules-sources.md`의 검수 6대 항목, source group, glossary 규칙을 기준으로 분류한다.
4. 표현 통일 판단에는 `/st-rag` 사례를 확인한다. RAG가 없으면 규칙/BX 기준의 권장이라고 명시한다.
5. 수정 필요이면 현재 문안, 제안, 쉬운 이유, 근거를 제시하되 Excel에는 적용하지 않는다.

## Rules

- 규칙 위반과 스타일 선호를 분리해서 말한다.
- `Needs Revision`은 반드시 수정이 아니며, `false positive` 또는 `유지`가 될 수 있다.
- 이미 사용자가 반영한 수정은 현재값을 재검증하고, 이전 문안을 다시 수정안으로 제시하지 않는다.
- 근거가 부족하면 불확실성을 명시한다.
- Excel 수정은 사용자가 별도 승인하기 전까지 하지 않는다.
