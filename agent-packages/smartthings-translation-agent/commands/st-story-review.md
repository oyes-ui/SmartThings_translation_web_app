---
description: AI 검수 결과를 source group·용어집·RAG·story 맥락으로 재평가 (크레딧 0)
argument-hint: <xlsx 경로> --sheet "BR(브라질)" [--ai-review <검수 결과 경로>] [--story story_049]
---

AI 최종평가를 후보 목록으로 받아, 대상 언어의 실제 현재 셀을 기준으로 story 단위 검수를 수행한다. 원본 Excel은 수정하지 않는다.

## 시작 안내

명령을 실행하면, 검수 전에 아래 절차와 현재 입력에서 확인된 정보를 먼저 짧게 안내한다.

1. **준비**: 대상 시트, story_id, AI 검수 결과 유무를 확인한다.
2. **기준 확정**: source group과 source 시트, glossary bracket occurrence를 확인한다.
3. **AI 재평가**: `Needs Revision`과 `Good` 코멘트를 모두 현재 셀 기준으로 확인한다.
4. **일관성 판단**: 표현 통일이 필요한 항목만 RAG로 확인하고, story의 호칭·주어·조사/격·어미·접속 표현을 점검한다.
5. **마무리**: `수정 필요 / 유지 / false positive / 추가 확인`으로 정리한 뒤, 필요하면 `/st-obsidian-report`에 반영한다.

입력이 부족하면 먼저 필요한 경로 또는 시트를 한 번에 요청한다. AI 검수 결과가 없으면 1~2단계와 story 맥락 검수만 진행한다고 안내한다.

## Workflow

1. `workbook_inspect.py`로 대상 시트와 섹션을 읽어 현재값, story_id, title/description 구조를 확인한다.
2. 대상 시트의 source group을 확정한다: KR source는 `KR → JA/CN/TW/US`, US source는 `US → BR/RU/DE…`.
3. AI의 `Needs Revision`뿐 아니라 `Good`의 코멘트·수정안도 후보로 수집한다.
4. source 원문의 bracket occurrence와 실제 glossary를 대조한다. 브랜드, 제목, bracket 없는 일반명사는 occurrence별로 분리한다.
5. 표현 통일이 쟁점이면 `/st-rag`로 같은 source group·대상 언어·콘텐츠 유형 사례를 조회한다.
6. story 전체의 호칭, 주어, 조사·전치사·격, 어미·활용, 접속 표현, 케이스, 불필요한 하이라이트를 검토한다.
7. 같은 source 개념이 story 안에서 다른 일반어로 번역됐으면(RAG와 별개로) 문장 역할이 동일한 범위에서 통일한다. 단, 현지 문법상 역할이 다른 표현은 기계적으로 맞추지 않는다.
8. 각 항목을 `수정 필요 / 유지 / false positive / 추가 확인`으로 재분류하고 `references/response-patterns.md`의 I 템플릿으로 보고한다.

## Rules

- AI 판정과 최종 판단을 반드시 분리한다. `Needs Revision`이 자동으로 수정 필요를 뜻하지 않는다.
- 이미 반영된 문안은 현재 셀 기준으로 검증하며, 과거 문안을 다시 제안하지 않는다.
- 수정 판단의 기준은 **현지화 자연스러움과 story 내부 일관성**이며, 용어·서식 규칙이 이를 우선 보완한다. 변경 범위는 이 기준을 충족하는 데 필요한 수준으로 정하고, 전체 문안을 제시할 때도 실제 변경 토큰을 분리해 설명한다.
- title source에 bracket이 없고 실제 UI 문자열 고정 근거가 없으면, 기능명을 현지 언어의 자연스러운 title로 풀어쓸 수 있다.
- 수정 제안은 `현재 → 제안 → 쉬운 이유 → 근거` 순서로 쓴다.
- 표현 통일은 RAG 근거를 우선한다. RAG가 없으면 규칙/BX 기준의 권장으로 표기한다.
- glossary 강제 규칙과 문법 규칙은 RAG보다 우선한다. Excel 수정은 별도 승인 후 `/st-story-apply`에서 납품 scope 전체 하이라이트·검증과 함께 수행한다. `/st-edit`는 저수준 임시 편집용이다.

## Story 049 검증 예시

- `CN(중국)`: KR source를 선택한다. AI가 구두점·bracket을 지적했더라도 실제 full-width punctuation, source occurrence, 문어체와 `您` 호칭의 문장 내 일관성을 다시 확인한다.
- `BR(브라질)`: US source를 선택한다. description에서 앱 명칭을 통일할 필요가 있으면 description RAG를 우선하고 disclaimer/navigation 사례와 구분한다.
- `RU(러시아)`: US source를 선택한다. title에는 glossary bracket/guillemet을 넣지 않고, 문장 안의 source-bracket occurrence만 용어 표기 규칙으로 재확인한다. AI의 과거 제안이 이미 반영됐으면 현재 셀을 유지로 기록한다.
