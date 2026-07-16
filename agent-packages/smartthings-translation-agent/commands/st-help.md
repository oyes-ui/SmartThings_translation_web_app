---
description: SmartThings 번역 에이전트 개요·검수 포인트·명령 안내
---

`agent-packages/smartthings-translation-agent/SKILL.md` 와 `agent-packages/smartthings-translation-agent/commands/` 를 근거로, 이 프로젝트를 처음 보는 사람도 이해할 수 있게 **한국어로** 다음을 간결히 출력하라:

1. **프로젝트 한 줄 소개**: SmartThings 다국어 번역·검수를 돕는 app-aware 에이전트. 앱(`src/translation_web_app/`)의 규칙·RAG·용어집·Excel 파이프라인을 래퍼 스크립트로 호출(재구현 안 함). 동작 수준 Level 1~4(Excel만 → offline RAG → semantic RAG → full pipeline).
2. **두 가지 실행 모드**: (a) **셀프 모드(크레딧 0)** — 에이전트가 `prompt_preview.py`+용어집+RAG(offline)로 직접 번역/검수. 소량·단건 기본값. (b) **파이프라인 모드(LLM 크레딧)** — `workbook_translate.py`/`workbook_audit.py`, `--pipeline`+승인 필요. (→ `agent-packages/smartthings-translation-agent/references/self-vs-pipeline.md`)
3. **검수 포인트**: audit 6항목(① 문법/자연스러움 ② 의미 충실 ③ 용어집 준수 ④ 현지화 ⑤ 대소문자 ⑥ 포맷·BX), story 문장 요소 일관성(호칭·주어·조사/격·어미·접속 표현), title↔description 맥락, 안전 규칙(원본 Excel 불변 / 수정 전 승인 / 크레딧 사전확인 / 시크릿 미노출).
4. **권장 검수 절차**: 아래 5단계를 짧게 안내한다.
   1. `/st-glossary-filter`: 대상 언어의 source group과 bracket occurrence를 확정한다.
   2. `/st-story-review`: AI 판정을 후보로 두고 실제 셀·용어집·문장 요소를 재평가한다.
   3. `/st-rag`: 표현 통일이 쟁점인 항목만, 같은 source group·콘텐츠 유형 사례로 확인한다.
   4. `/st-sections`: title-description 및 story 전체의 호칭·주어·조사/격·어미를 점검한다.
   5. `/st-story-apply`: 승인된 수정안으로 납품 scope 전체를 재하이라이트·검증한 최종 복사본을 만든다.
   6. `/st-obsidian-report`: `.delivery.json`을 근거로 언어별 반영 상태와 최종본 경로를 리포트에 갱신한다.
   AI `Good`/`Needs Revision`은 모두 후보이며, 최종 판단은 실제 셀·source group·용어집·RAG·언어 규칙으로 한다.
5. **명령 선택 가이드**: 전체 재평가는 `/st-story-review`, 특정 AI 판정의 이유는 `/st-audit-explain`, 과거 표현 근거는 `/st-rag`, 저수준 임시 편집은 `/st-edit`, 납품용 Excel 반영은 `/st-story-apply`이라고 한 줄로 구분한다.
6. **명령 목록**(크레딧 표시): `/st-help`(0) `/st-setup`(0) `/st-rules`(0) `/st-prompt`(0) `/st-glossary`(0) `/st-glossary-filter`(0) `/st-inspect`(0) `/st-story-review`(0) `/st-sections`(0) `/st-highlight`(0) `/st-story-apply`(0) `/st-textbook`(0) `/st-rag`(0~) `/st-ragdb`(0/빌드시 크레딧) `/st-edit`(0) `/st-translate`(LLM) `/st-audit`(LLM) `/st-audit-explain`(0) `/st-review-summary`(0) `/st-notebooklm`(외부 MCP) `/st-obsidian-report`(0). 각 한 줄 설명.

길게 늘어놓지 말고 스캔 가능한 요약으로.
