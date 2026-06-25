---
description: 앱과 동일한 번역/검수 프롬프트 생성 — 셀프 모드(크레딧 0)
argument-hint: <원문> --target-lang <시트> [--audit --translated <번역문>]
---

**셀프 모드 핵심.** 앱과 동일한 프롬프트를 만들어 **네(에이전트)가 직접** 번역/검수하라. LLM 호출 0.

```bash
python agent-packages/smartthings-translation-agent/scripts/prompt_preview.py --json $ARGUMENTS
```
번역: `--text "<원문>" --target-lang "DE(독일)" --row-key description`
검수: `--audit --text "<원문>" --translated "<번역문>" --target-lang "DE(독일)"`
용어·사례가 필요하면 `/st-glossary list` 와 `/st-rag` 결과를 함께 반영. 참조: `agent-packages/smartthings-translation-agent/references/self-vs-pipeline.md`
