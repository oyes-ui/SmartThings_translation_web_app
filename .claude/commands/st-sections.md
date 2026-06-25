---
description: 섹션 title↔description 맥락 일치 검토 (크레딧 0)
argument-hint: <xlsx 경로> [--sheet "JA(일본)"]
---

섹션별 title 이 description 의 핵심 benefit 을 반영하는지 검토한다.

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_inspect.py $ARGUMENTS --sections --json
```
결과는 `agent-packages/smartthings-translation-agent/references/response-patterns.md`(C-2 템플릿)로 제안. 수정은 승인 후 `/st-edit`.
