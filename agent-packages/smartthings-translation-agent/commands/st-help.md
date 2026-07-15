---
description: SmartThings 번역 에이전트 개요·검수 포인트·명령 안내
---

`agent-packages/smartthings-translation-agent/SKILL.md` 와 `agent-packages/smartthings-translation-agent/commands/` 를 근거로, 이 프로젝트를 처음 보는 사람도 이해할 수 있게 **한국어로** 다음을 간결히 출력하라:

1. **프로젝트 한 줄 소개**: SmartThings 다국어 번역·검수를 돕는 app-aware 에이전트. 앱(`src/translation_web_app/`)의 규칙·RAG·용어집·Excel 파이프라인을 래퍼 스크립트로 호출(재구현 안 함). 동작 수준 Level 1~4(Excel만 → offline RAG → semantic RAG → full pipeline).
2. **두 가지 실행 모드**: (a) **셀프 모드(크레딧 0)** — 에이전트가 `prompt_preview.py`+용어집+RAG(offline)로 직접 번역/검수. 소량·단건 기본값. (b) **파이프라인 모드(LLM 크레딧)** — `workbook_translate.py`/`workbook_audit.py`, `--pipeline`+승인 필요. (→ `agent-packages/smartthings-translation-agent/references/self-vs-pipeline.md`)
3. **검수 포인트**: audit 6항목(① 문법/자연스러움 ② 의미 충실 ③ 용어집 준수 ④ 현지화 ⑤ 대소문자 ⑥ 포맷·BX), story 문장 요소 일관성(호칭·주어·조사/격·어미·접속 표현), title↔description 맥락, 안전 규칙(원본 Excel 불변 / 수정 전 승인 / 크레딧 사전확인 / 시크릿 미노출).
4. **권장 검수 순서**: `/st-glossary-filter` → `/st-story-review` → 필요한 표현만 `/st-rag` → `/st-sections` → `/st-obsidian-report`. AI `Good`/`Needs Revision`은 모두 후보이며, 최종 판단은 실제 셀·source group·용어집·RAG·언어 규칙으로 한다.
5. **명령 목록**(크레딧 표시): `/st-help`(0) `/st-setup`(0) `/st-rules`(0) `/st-prompt`(0) `/st-glossary`(0) `/st-glossary-filter`(0) `/st-inspect`(0) `/st-story-review`(0) `/st-sections`(0) `/st-highlight`(0) `/st-textbook`(0) `/st-rag`(0~) `/st-ragdb`(0/빌드시 크레딧) `/st-edit`(0) `/st-translate`(LLM) `/st-audit`(LLM) `/st-audit-explain`(0) `/st-review-summary`(0) `/st-notebooklm`(외부 MCP) `/st-obsidian-report`(0). 각 한 줄 설명.

길게 늘어놓지 말고 스캔 가능한 요약으로.
