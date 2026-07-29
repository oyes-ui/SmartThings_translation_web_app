---
description: 읽기 전용 통합 번역 검수와 Markdown 리포트·수정 제안 생성
argument-hint: <xlsx 경로> [--sheet <시트>]
---

`/st-review`는 Excel을 수정하지 않는다. 기존 `st-inspect`, `st-sections`, `st-story-review`,
`st-rag`, `st-obsidian-report`, `st-review-summary`를 상황에 맞게 조합한다.

출력은 `report_format_spec.md` 계약을 따르는 Markdown 리포트와 수정 제안이다. 각 제안에는
sheet/cell, before/after, rule IDs, RAG 근거 또는 보류 사유를 남긴다. 제안 확정 뒤에는 아래 읽기 전용 생성기로 공통 Markdown 리포트와 `pending_approval` manifest를
고정한다. 이 명령은 workbook을 수정하지 않는다.

```bash
python scripts/review_report_builder.py story.xlsx proposals.json \
  --report-id review-YYYYMMDD-001 --source-file-id story-001 --output-dir outputs/review --json
```

사람이 manifest의 항목을 `approval_status: approved`로 바꾼 뒤에만 `/st-apply`가 이를 처리한다.
