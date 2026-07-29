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

### story 단위 self audit (크레딧 0)

워크북 전체를 LLM 파이프라인에 돌리기 전에, 아래 순서로 크레딧 0 self audit을 먼저 수행한다(→ `commands/st-audit.md`):

1. `workbook_inspect.py --sections`로 source와 target의 모든 section을 그룹째 뽑고 workbook의 실제 source group을 확정한다. 표준 story mapping의 `JA/TW → KR`은 적용하되, 개별 workbook의 선언이 우선한다.
2. 기존 AI 후보와 무관하게 각 언어의 story 콘텐츠 셀(C7·C8·C10·C11·C13·C15·C16·C17)을 source↔target으로 **독립 재독해**한다. 의미 주체, 직역투, title 역할, section 연결, 용어/서식을 함께 점검한다.
3. 독립 재독해와 AI 후보 재판정에서 좁혀진 셀만 `prompt_preview.py --audit`으로 개별 확인한다. `prompt_preview --audit`은 후보 확인 도구이며 story 전체 읽기를 대체하지 않는다.
4. 표현 통일이 실제 쟁점인 경우에만 RAG를 사용한다. RAG 사례가 없더라도 문법·의미·직역투는 규칙/BX 기준으로 판단한다.
5. 수정 수가 적을수록 독립 재점검을 생략하지 않고, 기존 후보 누락 여부를 더 엄격히 점검한다.
6. 대량·자동화·재현 가능한 산출물이 꼭 필요할 때만 승인 후 `--pipeline`으로 넘어간다.

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
