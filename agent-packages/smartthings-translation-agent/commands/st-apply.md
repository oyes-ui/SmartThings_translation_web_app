---
description: 승인 manifest 기반으로 감수본·납품본을 새 Excel 사본에 반영
argument-hint: <review-workbook.xlsx> <approval-manifest.json>
---

`/st-apply`는 최종 납품 경로다. 기존 `st-review-apply`, `st-story-apply`, `st-highlight`의
검증 책임을 통합한다.

- 사람 승인 manifest의 항목만 반영한다.
- 원본은 절대 덮어쓰지 않는다.
- delivery scope와 필요한 KR/US source sheet를 glossary 기준으로 재하이라이트한다.
- 실제 diff, 보호 시트, 텍스트 보존, highlight report를 검증한다.

## 실행 경로

`report_format_spec.md`의 표준 검수 승인 manifest(`manifest_schema_version: 1` 또는 `2`,
`approval_status: approved`)는 `workbook_review_apply.py`가 직접 `decisions`로 변환한다.
`approved` 외 상태는 Excel에 반영되지 않으며, 이 경로는 native review와 agent review 모두에
공통으로 사용한다.

```bash
python scripts/workbook_review_apply.py review.xlsx approval-manifest.json \
  --output story_accepted.xlsx --glossary Glossary.csv --app-root /path/to/app --json
```

live layer가 만든 manifest는 반드시 `approval: "approved"`와 각 변경의
`verification: "verified"`를 가져야 한다. `preview`나 `fallback_delivery` 항목은 적용되지 않는다.

```bash
# 1) manifest 형식/승인 상태 검증 — 파일을 쓰지 않음
python scripts/excel_live_manifest.py approval-manifest.json --require-approved --json

# 2) 일반 draft 복사본 — 원본 불변, 값/구조/full diff 검증
python scripts/workbook_apply_edits.py story.xlsx approval-manifest.json --json

# 3) 납품본 — delivery scope와 KR/US source sheet 전체 glossary rich-text 재생성/검증
python scripts/workbook_story_apply.py story.xlsx approval-manifest.json \
  --delivery-sheets "CO(콜롬비아)" --glossary Glossary.csv --app-root /path/to/app --json
```

원어민 감수의 `accept`/`partial` 판정 manifest는 기존 `/st-review-apply`를 계속 사용한다. live
manifest는 `before`/`after`가 확정된 일반 편집·콜롬비아 현지화 수정의 handoff 계약이다.

## 검토 결과 기록 (자율화 근거 축적)

승인/거절을 마친 manifest는 Excel에 반영하기 전후로 **반드시 outcome ledger에 기록한다.** 사람이
이미 내린 판단을 그대로 남기는 것이며 추가 작업이 아니다. 이 기록이 없으면 에이전트 제안의
정확도를 측정할 수 없고, 승인 게이트를 완화할 근거도 영원히 쌓이지 않는다.

```bash
python scripts/review_outcomes.py --ledger outputs/review/outcomes.jsonl \
  --manifest approval-manifest.json \
  --emit-golden outputs/review/golden.json --emit-results outputs/review/results.json

python scripts/quality_scorecard.py outputs/review/golden.json outputs/review/results.json
```

- 거절한 항목은 `approval_status`를 `approved` 외의 값으로 두고 `rejection_reason`을 적는다.
  거절도 승인만큼 중요한 근거다.
- 문안을 고쳐서 승인한 경우 `after`만 바꾸고 `proposed_after`는 건드리지 않는다. 이 두 값의
  차이가 "에이전트가 얼마나 근접했는지"를 재는 유일한 근거다.
- 독립성 경고(`anchoring`)가 붙은 근거에 기댄 항목은 `independence: suspect`로 표시되어 기본적으로
  점수 계산에서 제외된다. 가짜 합의로 정확도가 부풀려지는 것을 막기 위해서다.
- `review_outcomes.py`는 측정만 한다. 어떤 항목도 자동 승인하거나 적용하지 않는다.

## Obsidian 상태 반영

`/st-apply` 자체는 vault를 쓰지 않는다. 사용자가 명시적으로 요청한 경우에만 성공 result manifest를
근거로 workspace Obsidian 초안의 상태를 갱신한다. manifest가 없거나 검증 실패면 `applied`로 기록하지 않는다.

```bash
python scripts/obsidian_workflow.py sync-status outputs/obsidian/review-001.md \
  story_final.review_apply.json --output outputs/obsidian/review-001-applied.md
```
