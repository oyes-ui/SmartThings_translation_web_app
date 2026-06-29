---
description: 앱 검수(inspection) 파이프라인 실행 [LLM 크레딧]
argument-hint: <xlsx 경로> --sheets <대상>
---

⚠️ **LLM 크레딧 소모.** 소량·단건이면 먼저 `/st-prompt --audit`(셀프, 크레딧 0)를 권하라. 사용자 승인 후에만 `--pipeline` 으로 실행:

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_audit.py $ARGUMENTS --pipeline --json
```
`--pipeline` 없으면 거부+셀프 안내. `.audit_report.txt` 생성. 참조: `agent-packages/smartthings-translation-agent/references/self-vs-pipeline.md`
