---
description: app repo 연결 + 동작 레벨 점검(bootstrap)
argument-hint: [app-root 경로]
---

app repo 를 연결하고 동작 레벨(1~4)을 점검하라. 인자로 경로가 오면 `--app-root` 로 전달한다.

```bash
python agent-packages/smartthings-translation-agent/scripts/bootstrap.py --app-root "$ARGUMENTS" --json
```
경로가 비었으면 저장 config/자동탐색을 쓴다. 재방문 고정이 필요하면 `--save` 안내.
참조: `agent-packages/smartthings-translation-agent/references/setup-workflow.md`
