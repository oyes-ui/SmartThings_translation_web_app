---
description: 구조화 텍스트 → source 워크북 생성 (크레딧 0)
argument-hint: --spec <json파일> | --spec-json '<json>'
---

story(title/description)+최대 4 section(title/description/disclaimer/button)을 템플릿에 채워 source 워크북을 만든다.

```bash
python agent-packages/smartthings-translation-agent/scripts/text_workbook_create.py $ARGUMENTS --json
```
spec 스키마는 스크립트 docstring 참조. source_sheet(예: "US(미국)") 필수.
