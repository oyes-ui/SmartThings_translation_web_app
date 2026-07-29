---
description: SmartThings 규칙·용어집·RAG 사례를 안전하게 질의
argument-hint: <질문 또는 표현>
---

`/st-ask`는 읽기 전용 지식 질의다. 질문의 성격에 맞게 기존 구현 도구를 선택한다.

- 규칙/locale: `st-rules`와 Markdown rules source
- 용어집: `st-glossary`, 필요 시 `st-glossary-filter`
- 과거 사례: `st-rag` (`exact`/`keyword`/`semantic` 구분)
- 프롬프트 확인: `st-prompt`

명시 규칙·glossary·승인된 시장 기준이 RAG보다 우선임을 표시한다. 쓰기, glossary CRUD,
RAG DB build, API 비용이 드는 작업은 이 명령에서 실행하지 않는다.
