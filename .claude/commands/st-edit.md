---
description: 승인된 셀 편집을 복사본에 적용 (원본 불변, 크레딧 0)
argument-hint: <xlsx 경로> '<edits JSON>'
---

**사용자 승인을 받은** 셀 편집만 적용한다. 원본은 그대로 두고 타임스탬프 복사본 생성. 승인 없이 실행 금지.

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_apply_edits.py $ARGUMENTS
```
edits 예: `'[{"sheet":"JA(일본)","cell":"C10","new_value":"..."}]'`. 참조: `agent-packages/smartthings-translation-agent/references/excel-workflow.md`
