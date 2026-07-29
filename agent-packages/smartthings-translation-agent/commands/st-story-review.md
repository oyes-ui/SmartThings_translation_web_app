---
description: AI 후보 재판정 + 후보 비의존 독립 story 재검수 (크레딧 0)
argument-hint: <xlsx 경로> --sheet "BR(브라질)" [--ai-review <검수 결과 경로>] [--story story_049]
---

AI 최종평가를 **1차 후보 목록**으로만 사용하고, 대상 언어의 실제 현재 셀을 기준으로 후보 비의존 2차 재독해까지 수행한다. 원본 Excel은 수정하지 않는다.

## 시작 안내

명령을 실행하면, 검수 전에 아래 절차와 현재 입력에서 확인된 정보를 먼저 짧게 안내한다.

1. **준비**: 대상 시트, story_id, AI 검수 결과 유무를 확인한다.
2. **기준 확정**: source group과 source 시트, glossary bracket occurrence를 확인한다.
3. **1차 후보 재판정**: `Needs Revision`과 `Good` 코멘트를 모두 현재 셀 기준으로 확인한다.
4. **2차 독립 재점검**: AI TXT를 근거에서 제외하고 source↔target의 전체 story 셀을 재독해한다.
5. **마무리**: 두 pass를 분리해 `수정 필요 / 유지 / false positive / 추가 확인`으로 정리한 뒤, 필요하면 `/st-obsidian-report`에 반영한다.

입력이 부족하면 먼저 필요한 경로 또는 시트를 한 번에 요청한다. AI 검수 결과가 없으면 1~2단계와 story 맥락 검수만 진행한다고 안내한다.

## 필수 2-pass Workflow

### 공통 준비

1. `workbook_inspect.py --sections`로 대상 시트와 source 시트를 읽어 현재값, story_id, title/description 구조를 확보한다.
2. **source group을 먼저 잠근다.** 현재 표준 story mapping에서는 `JA/TW`가 KR source, 그 밖의 target이 US source이지만, workbook의 실제 source-group 선언이 우선이다. source가 명시한 UI 활성화·호환 조건은 다른 source group의 문구로 교체하지 않는다.
3. source 원문의 bracket occurrence와 실제 glossary를 대조한다. 브랜드, title, bracket 없는 일반명사는 occurrence별로 분리한다.

### Pass 1 — AI 후보 재판정

4. AI의 `Needs Revision`뿐 아니라 `Good`의 코멘트·수정안도 후보로 수집하고, 현재 셀 기준으로 재분류한다.
5. 표현 통일이 실제 쟁점인 경우에만 `/st-rag`로 같은 source group·대상 언어·콘텐츠 유형 사례를 확인한다.

### Pass 2 — 후보 비의존 독립 재점검

6. AI TXT의 판정·수정안을 근거에서 제외하고, **각 대상 언어의 C7·C8·C10·C11·C13·C15·C16·C17 전체**를 source↔target으로 다시 읽는다.
7. 아래 체크리스트를 모두 확인한다.
   - **의미 주체:** 누가 누구를 그리워하거나 돌보는지, 능동/수동 및 대상 관계가 바뀌지 않았는지.
   - **문장 자연스러움:** 관용구 직역, 부자연스러운 연결·전치사·격·어미, 마케팅 카피 리듬.
   - **story 역할:** C7/C10처럼 source가 구별한 상황·혜택·기능 초점이 평탄화되지 않았는지. 문자열 완전 일치 검사만으로 통과시키지 않는다.
   - **section 연결:** title → description → CTA → disclaimer의 지칭·기능명·사용자 혜택·조건이 이어지는지.
   - **용어·서식:** bracket occurrence, navigation path 예외, 브랜드, UI 라벨, source-confirmed 기능 조건.
8. 같은 source 개념이 같은 story 역할에서 다른 일반어로 번역됐다면 통일 후보로 기록한다. 단, 현지 문법상 역할이 다른 표현은 기계적으로 맞추지 않는다.

### 종료 게이트 및 보고

9. 언어별 독립 재점검 결과가 기록되지 않으면 `추가 수정 없음`, Fix 생성, 전체 통과 결론을 내릴 수 없다.
10. `후보 기반 발견 수 / 독립 발견 수 / 유지 언어 수 / 언어별 완료 상태`를 먼저 집계한다.
11. 각 항목을 `수정 필요 / 유지 / false positive / 추가 확인`으로 재분류하고 `references/response-patterns.md`의 I 템플릿으로 보고한다. RAG 부재는 문법·의미·직역투 판단을 보류하는 근거가 아니다.

셀 단위의 번역·검수 후보를 재평가해야 할 때는 `prompt_preview.py`로 앱과 동일한 언어별 규칙을 확인한다. 표준 워크북 시트명(예: `CN(중국)`, `DE(독일)`, `JA(일본)`)은 `PromptBuilder`에서 canonical 언어명으로 자동 정규화되므로, 시트명을 그대로 `--target-lang`에 사용해도 된다.

## Rules

- AI 판정과 최종 판단을 반드시 분리한다. `Needs Revision`이 자동으로 수정 필요를 뜻하지 않는다.
- 이미 반영된 문안은 현재 셀 기준으로 검증하며, 과거 문안을 다시 제안하지 않는다.
- 수정 판단의 기준은 **현지화 자연스러움과 story 내부 일관성**이며, 용어·서식 규칙이 이를 우선 보완한다. 변경 범위는 이 기준을 충족하는 데 필요한 수준으로 정하고, 전체 문안을 제시할 때도 실제 변경 토큰을 분리해 설명한다.
- title source에 bracket이 없고 실제 UI 문자열 고정 근거가 없으면, 기능명을 현지 언어의 자연스러운 title로 풀어쓸 수 있다.
- 수정 제안은 `현재 → 제안 → 쉬운 이유 → 근거` 순서로 쓴다.
- 표현 통일은 RAG 근거를 우선한다. RAG가 없으면 규칙/BX 기준의 권장으로 표기한다.
- glossary 강제 규칙과 문법 규칙은 RAG보다 우선한다. Excel 수정은 별도 승인 후 `/st-story-apply`에서 납품 scope 전체 하이라이트·검증과 함께 수행한다. `/st-edit`는 저수준 임시 편집용이다.
- 자동 Fix 후에는 값 diff·보호 언어 무결성·전체 재하이라이트뿐 아니라, 독립 재점검에 기록한 수정 셀과 실제 값 diff가 정확히 일치하는지 검증한다.

## 039 회귀 점검 기준

- BE/CA/IT C8처럼 source의 의미 주체가 역전된 문장을 Pass 2에서 잡아야 한다.
- ES C11/C16처럼 후보에 없던 현지화 직역투를 Pass 2에서 후보로 기록해야 한다.
- NL/AE/TR/PL/TH의 C7/C10처럼 문자열이 달라도 `away`/`out` 역할이 평탄화된 title을 story 역할 오류로 판정해야 한다.
- SG C8의 `worry on the side`처럼 관용구 어순이 어색한 target을 자연스러움 오류로 판정해야 한다.
- JA/TW C17의 `Now brief 활성화`는 KR source의 UI 조건과 일치하면 false positive로 유지해야 한다.

## Story 049 검증 예시

- `CN(중국)`: KR source를 선택한다. AI가 구두점·bracket을 지적했더라도 실제 full-width punctuation, source occurrence, 문어체와 `您` 호칭의 문장 내 일관성을 다시 확인한다.
- `BR(브라질)`: US source를 선택한다. description에서 앱 명칭을 통일할 필요가 있으면 description RAG를 우선하고 disclaimer/navigation 사례와 구분한다.
- `RU(러시아)`: US source를 선택한다. title에는 glossary bracket/guillemet을 넣지 않고, 문장 안의 source-bracket occurrence만 용어 표기 규칙으로 재확인한다. AI의 과거 제안이 이미 반영됐으면 현재 셀을 유지로 기록한다.
