---
description: 앱 번역(+검수) 파이프라인 실행 [LLM 크레딧]
argument-hint: <xlsx 경로> --sheets <대상> [--translate-only]
---

⚠️ **LLM 크레딧 소모.** 소량·단건이면 먼저 `/st-prompt`(셀프, 크레딧 0)를 권하라. 사용자가 대량/자동화를 원하고 **명시적으로 승인**했을 때만 `--pipeline` 을 붙여 실행:

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_translate.py $ARGUMENTS --pipeline --json
```
`--pipeline` 없으면 스크립트가 거부하고 셀프 모드를 안내한다. 참조: `agent-packages/smartthings-translation-agent/references/self-vs-pipeline.md`
