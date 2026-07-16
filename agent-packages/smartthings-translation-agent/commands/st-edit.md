---
description: 승인된 셀 편집을 임시 복사본에 적용 (원본 불변, 크레딧 0)
argument-hint: <xlsx 경로> '<edits JSON>'
---

**사용자 승인을 받은** 셀 편집만 적용한다. 원본은 그대로 두고 타임스탬프 복사본을 생성한다. rich text 하이라이트는 보존되지 않을 수 있으므로, 이 명령의 산출물은 단독 납품본으로 안내하지 않는다.

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_apply_edits.py $ARGUMENTS
```
edits 예: `'[{"sheet":"JA(일본)","cell":"C10","new_value":"..."}]'`. 참조: `agent-packages/smartthings-translation-agent/references/excel-workflow.md`

story 검수에서 확정된 수정안을 납품용으로 만들 때는 `/st-story-apply`를 사용한다. 해당 명령은 delivery scope 전체 재하이라이트와 값 변경 검증까지 완료한다.
