---
description: 읽기 전용 통합 번역 검수와 Markdown 리포트·수정 제안 생성
argument-hint: <xlsx 경로> [--sheet <시트>]
---

`/st-review`는 Excel을 수정하지 않는다. 기존 `st-inspect`, `st-sections`, `st-story-review`,
`st-rag`, `st-obsidian-report`, `st-review-summary`를 상황에 맞게 조합한다.

출력은 `report_format_spec.md` 계약을 따르는 Markdown 리포트와 수정 제안이다. 각 제안에는
sheet/cell, before/after, rule IDs, RAG 근거 또는 보류 사유를 남긴다. 적용은 사람 승인 후
`/st-apply` 또는 일반 수정이면 `/st-edit`에서만 수행한다.
