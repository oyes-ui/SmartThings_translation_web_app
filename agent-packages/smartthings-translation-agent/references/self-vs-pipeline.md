# 셀프 모드 vs 파이프라인 모드 (번역·검수)

번역과 검수는 **두 가지 경로**로 할 수 있다. 기본은 **셀프 모드(크레딧 0)** 다.

## 셀프 모드 — 에이전트가 직접 (크레딧 0, 기본)

에이전트(Claude) 자체가 LLM이므로, 앱의 규칙·용어집·과거 사례를 받아 **직접** 번역/검수한다.
별도 Gemini/GPT 호출이 없으므로 LLM 크레딧이 0이다.

핵심 도구: `scripts/prompt_preview.py` — 앱의 `PromptBuilder` 를 그대로 호출해, 앱이 모델에 보낼 것과
**동일한** 프롬프트(페르소나·공통 현지화·언어별 규칙·BX·타이포·용어집 포맷·RAG)를 만든다.

권장 셀프 워크플로:
1. **규칙 프롬프트 확보**
   ```bash
   python scripts/prompt_preview.py --text "<원문>" --target-lang "<시트>" --row-key <맥락> [--bx] [--glossary]
   ```
2. **용어집 확인** (해당 원문에 걸리는 용어/규칙)
   ```bash
   python scripts/glossary_manage.py list --search "<핵심어>" --json
   ```
3. **과거 사례 확보** (일관성)
   ```bash
   python scripts/rag_lookup.py --query "<원문>" --target-lang <코드> --json   # 키 없으면 offline
   ```
   필요 시 그 결과를 `prompt_preview.py --rag-context "<문자열>"` 로 다시 주입해 프롬프트에 포함.
4. **에이전트가 직접 번역/검수** — 위 프롬프트·용어·사례를 근거로 결과를 산출하고, 검수는
   `references/response-patterns.md` 의 6항목 템플릿으로 설명한다.

검수도 동일: `prompt_preview.py --audit --text "<원문>" --translated "<번역문>" --target-lang "<시트>"`.

언제 셀프 모드인가: **소량·단건·대화형**, 키가 없을 때, 빠른 검토/설명이 필요할 때. → 기본값.

## 파이프라인 모드 — 앱의 유료 LLM (LLM 크레딧)

대량 셀을 일괄 처리하거나, 앱과 동일한 자동 산출물(번역+검수 Excel/리포트)이 필요할 때만.

- 번역(+검수): `scripts/workbook_translate.py <xlsx> --pipeline [--translate-only] --sheets "<대상>"`
- 검수 전용: `scripts/workbook_audit.py <xlsx> --pipeline --sheets "<대상>"`

안전 장치:
- `--pipeline` 플래그가 없으면 스크립트가 **실행을 거부**하고 셀프 모드를 안내한다.
- 크레딧을 소모하므로 **실행 전 사용자 승인**을 받는다(안전 규칙 3).
- 원본 워크북은 수정하지 않는다(앱이 새 파일 생성).

언제 파이프라인 모드인가: **워크북 전체/여러 시트 자동화**, 재현 가능한 일괄 산출물, 역번역 등
앱 고유 처리가 필요할 때. 사용자가 명시적으로 원하고 승인했을 때만.

## 한눈 비교

| 항목 | 셀프 모드 | 파이프라인 모드 |
|---|---|---|
| LLM 크레딧 | 0 | 소모 |
| 도구 | `prompt_preview.py` (+glossary/rag) | `workbook_translate.py` / `workbook_audit.py` |
| 적합 | 소량·단건·대화·설명 | 대량·자동화·일괄 산출물 |
| 키 필요 | 불필요(offline RAG 시) | Gemini/GPT 키 필요 |
| 승인 | 일반 진행 | 실행 전 승인 + `--pipeline` |
