---
description: 번역하기 — 소량은 직접 번역, 대량 API 실행은 승인 후
argument-hint: <문구 또는 파일> <대상 언어와 요청>
---

`references/self-vs-pipeline.md`를 따른다. 사용자는 번역할 문구/파일과 언어를 자연어로 지정한다.

- 소량·단건은 `scripts/prompt_preview.py`로 앱 규칙·용어집·필요한 RAG 근거를 받고
  에이전트가 직접 번역한다. 추가 번역 API 호출은 없다.
- 번역안을 Excel 수정본으로 만들어 달라는 요청은 사용자에게 변경안을 보여주고 기존 승인
  범위 안에서 `/st-edit`의 계약 실행 경로로 반영한다. 최종 납품 요청은 `/st-apply`로 연결한다.
- 대량/자동화 API 번역을 명시적으로 승인한 경우에만 아래 기존 도구를 실행한다.
  대상 파일·언어·작업 범위와 비용 발생을 먼저 안내하며 비용 수치를 추정해서 확정하지 않는다.

```bash
python scripts/workbook_translate.py <workbook.xlsx> --sheets "<대상>" --pipeline --json
```

기존 `--translate-only` 등 API 옵션은 유지한다. API 실행/재시도를 local Excel 복구 루프로
자동 수행하지 않는다. 소량 번역 요청을 곧바로 API 실행으로 해석하지 않는다.

## 업무 전체를 이어갈 때

`references/workflow-guide.md`와 `references/workbook-batch.md`를 읽는다. 파일별 용어집 적용안을 확정한 뒤 `workbook_batch.py`로 기존 기능을 연결한다. 기본 API는 초벌·Excel 기입·하이라이트만이며 API 검수·역번역은 명시 요청 때만 실행한다. 초벌 뒤 활성 에이전트가 `ready_for_agent`의 셀→시트→리드 prompt를 수행하고 `advance`를 반복해 상세 MD·통합 승인검토표까지 생성한다. 준비 명령만 실행하고 검수가 완료됐다고 보고하지 않는다. 성공 작업은 계속하며 유료 실패는 자동 재시도하지 않는다.
