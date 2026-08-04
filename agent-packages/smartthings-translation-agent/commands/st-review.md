---
description: `/st-inspect` 언어 시트 에이전트 검수의 호환 alias
argument-hint: <xlsx 경로> [--sheet <시트>]
---

`/st-review`는 전환 기간의 호환 alias다. 새 검수는 `/st-inspect <xlsx> --sheet "<언어 시트>"`
로 실행한다. 기존 자동화·사용자 습관을 보호하기 위해 유지하며, 4주 shadow 평가 후 유지·폐기를
결정한다.

`/st-inspect`와 마찬가지로 Excel을 수정하지 않는다. semantic RAG 예산, 결정론적 glossary 근거
주입, 5개 서브에이전트 취합, `pending_approval` manifest 정책을 동일하게 따른다.

출력은 schema v2의 Markdown 리포트와 수정 제안이다. 기존 `changes[]` 계약은 유지한다. 각 제안에는
sheet/cell, before/after, rule IDs, RAG 근거 또는 보류 사유를 남긴다. 제안 확정 뒤에는 아래 읽기 전용 생성기로 공통 Markdown 리포트와 `pending_approval` manifest를
고정한다. 이 명령은 workbook을 수정하지 않는다.

```bash
python scripts/review_report_builder.py story.xlsx proposals.json \
  --report-id review-YYYYMMDD-001 --source-file-id story-001 --output-dir outputs/review --json
```

사람이 manifest의 항목을 `approval_status: approved`로 바꾼 뒤에만 `/st-apply`가 이를 처리한다.

Obsidian용 연결은 기본적으로 workspace 초안만 만든다. 사용자가 vault 발행 또는 과거 노트 검색을
명시한 경우에만 `/st-obsidian-report`와 `scripts/obsidian_workflow.py`를 사용한다.
