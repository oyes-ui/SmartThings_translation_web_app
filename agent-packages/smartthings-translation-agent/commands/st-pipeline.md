---
description: 승인 후 앱 LLM 번역 또는 검수 파이프라인 실행
argument-hint: translate|audit <xlsx 경로> --sheets <시트>
---

`/st-pipeline`은 비용이 드는 고급 경로다. 사용자가 명시적으로 승인한 경우에만 기존
`st-translate` 또는 `st-audit`를 `--pipeline`으로 호출한다.

- `translate`: 앱 번역(+선택적 검수) 파이프라인
- `audit`: 앱 inspection 파이프라인

실행 전 대상 workbook·sheet·예상 비용/영향을 확인하고, 원본 파일은 수정하지 않는다.
