---
description: 지정한 내용 수정하기 — 변경안 확인 후 수정본(draft)
argument-hint: <파일과 수정할 내용 또는 자연어 요청>
---

사용자가 지정한 셀/문구 수정을 다룬다. 검수의 승인안을 최종 납품하는 요청은 `/st-apply`다.
`references/excel-workflow.md`와 `references/command-execution.md`를 먼저 읽는다.

대화의 파일·수정 내용을 이용해 에이전트가 edits JSON을 준비한다. 사용자에게 JSON 작성을
요구하지 않는다. 현재→수정안과 적용 범위를 보여주고 승인된 내용만 복사본에 적용한다.
승인되지 않은 문안을 적용하지 않으며 기존 승인이 있으면 중복 승인을 요청하지 않는다.

```bash
# preview: workbook 쓰기 없음
python scripts/workbook_apply_edits.py <workbook.xlsx> <edits.json> --dry-run --json
# 명시적으로 승인한 수정 적용/동일 작업 재개: 계약 실행기가 기본 경로
python scripts/workbook_apply_edits.py <workbook.xlsx> <edits.json> --json
```

`before` 불일치, 수식/병합/보호/숨김 시트의 기존 예외 게이트를 유지한다. 원본 불변,
독립 5축 검증, 상태·이벤트·제한 복구·재개를 실행하며 계약 파일과 작업 경로는 내부에서 관리한다.
결과는 draft이고 최종 번역 납품본으로 안내하지 않는다. 납품 요청에는 `/st-apply`를 사용한다.

수정 문자 빨강+용어집 파랑 검수 표시는 고급 `workbook_incremental_highlight.py`를 사용한다.
이 경로도 계약·독립 5축 검증 후 버전 배치로 공개한다. 반환된 `output` 경로를 안내한다.
일반 편집과 무관하게 자동으로 추가 실행하지 않는다.
