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

`report_format_spec.md`의 표준 검수 승인 manifest(`manifest_schema_version: 1`,
`approval_status: approved`)는 `scripts/report_manifest_adapter.py`가 처리한다. `approved`가
아닌 항목은 결과의 `skipped`에 남고 적용되지 않는다. 이 manifest는 아래 draft/delivery 명령에
직접 전달할 수 있다.

```bash
# 표준 검수 manifest의 승인 항목 확인 — 파일을 쓰지 않음
python scripts/report_manifest_adapter.py approval-manifest.json --edits-only
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
