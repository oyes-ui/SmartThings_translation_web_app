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
sheet/cell, before/after, rule IDs, 지지 관점, RAG 근거 또는 보류 사유를 남긴다. 리포트와
`pending_approval` manifest는 병합 단계에서 함께 만들어진다. 이 명령은 workbook을 수정하지 않는다.

```bash
python scripts/agent_sheet_merge.py --packet packet.json --opinions-dir <역할별 의견서 dir> \
  --workbook story.xlsx --report-id review-YYYYMMDD-001 --output-dir outputs/review
```

`review_report_builder.py`는 라이브러리이며 더 이상 CLI로 직접 호출하지 않는다. 자유 형식
`proposals.json`을 받던 경로는 §4-A에 따라 제거됐다 — 리드 에이전트가 의견서를 요약해
`changes[]`를 만드는 우회를 막기 위해서다.

사람이 manifest의 항목을 `approval_status: approved`로 바꾼 뒤에만 `/st-apply`가 이를 처리한다.

Obsidian용 연결은 기본적으로 workspace 초안만 만든다. 사용자가 vault 발행 또는 과거 노트 검색을
명시한 경우에만 `/st-obsidian-report`와 `scripts/obsidian_workflow.py`를 사용한다.
