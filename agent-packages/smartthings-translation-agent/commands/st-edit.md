---
description: 일반 Excel 셀 편집 preview·승인·복사본 적용 (원본 불변, 크레딧 0)
argument-hint: <xlsx 경로> '<edits JSON>' [--dry-run]
---

`/st-edit`는 검수 승인과 별개의 일상 수정 경로다. 기본은 preview이며, 사용자 승인 뒤에만
원본이 아닌 타임스탬프 복사본을 만든다. 첫 실행에는 원본 옆 `.st-history/`에 기준 manifest를
기록하고, 승인 적용에는 셀별 before/after·문자 diff·revision ID를 기록한다. 일반 수정본은
`draft`이며 단독 납품본으로 안내하지 않는다.

```bash
# 1) preview — 파일을 쓰지 않음
python agent-packages/smartthings-translation-agent/scripts/workbook_apply_edits.py \
  <workbook.xlsx> '<edits JSON>' --dry-run --json

# 2) 사용자 승인 후 적용 — 원본은 불변
python agent-packages/smartthings-translation-agent/scripts/workbook_apply_edits.py \
  <workbook.xlsx> '<edits JSON>' --json
```
edits는 `before`와 `after`를 권장한다:

```json
[{"sheet":"JA(일본)","cell":"C10","before":"기존 문구","after":"승인된 문구"}]
```

적용 도구는 `before` 불일치, 수식 셀, 병합 범위, 보호/숨김 시트를 기본 중단한다. 해당
예외는 명시적 `--allow-*` 플래그와 사용자 승인 뒤에만 처리한다. 저장 뒤에는 대상값뿐 아니라
시트 순서·병합·보호·고정 창과 승인 범위 밖 값/수식까지 다시 비교한다. 검증 실패본은 delivery로
안내하지 않는다. 참조: `agent-packages/smartthings-translation-agent/references/excel-workflow.md`

story 검수나 감수 승인안을 납품용으로 만들 때는 `/st-apply`(현재 내부 호환 경로:
`/st-story-apply`, `/st-review-apply`)를 사용한다. 해당 경로만 delivery scope 전체
재하이라이트와 값 변경 검증까지 완료한다.

수정 문자(빨강)와 glossary term(파랑)을 함께 보는 검수본은 승인 적용 뒤 아래 고급 경로로 만든다.
glossary가 겹치는 문자에서는 파란색이 최종 우선이며, 같은 revision을 재실행하면 이미 렌더링된
셀은 건너뛴다.

```bash
python scripts/workbook_incremental_highlight.py <revised.xlsx> \
  --revision-manifest <revision.json> --glossary <Glossary.csv> \
  --sheets "CO(콜롬비아)" --app-root /path/to/app --json
```
