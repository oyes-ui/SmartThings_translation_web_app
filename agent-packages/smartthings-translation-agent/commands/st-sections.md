---
description: 섹션·story 맥락 및 문장 요소 일관성 검토 (크레딧 0)
argument-hint: <xlsx 경로> [--sheet "JA(일본)"]
---

섹션별 title이 description의 핵심 benefit을 반영하는지, story 전체의 문장 요소와 표기가 일관적인지 검토한다.

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_inspect.py $ARGUMENTS --sections --json
```
## 검토 범위

1. title↔description의 benefit, 기능명, 톤 정합성
2. story 전체의 호칭, 주어, 조사·전치사·격, 어미·활용, 접속 표현, 강조 부사 반복
3. 대소문자, 불필요한 glossary 하이라이트/대괄호, 제목의 용어 표기 예외
4. 표현 통일 판단 전 대상 시트의 source group을 확정한다: KR source는 `KR → JA/CN/TW/US`, US source는 `US → BR/RU/DE…`.
5. 표현 통일이 필요한 경우 RAG 사례를 먼저 조회한다. glossary 강제 규칙과 문법 규칙은 RAG보다 우선한다.

언어 규칙이 있는 경우 canonical rule을 먼저 확인한다. 예: CN은 문어체와 명시적 존칭의 일관성, RU는 제목의 glossary 괄호 금지 및 문장 내 용어 표기, BR은 설명문 RAG에 근거한 앱 명칭 일관성을 검토한다.

결과는 `agent-packages/smartthings-translation-agent/references/response-patterns.md`(C-2 템플릿)로 제안한다. 수정은 승인 후 `/st-edit`.
