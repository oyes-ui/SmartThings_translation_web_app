---
description: 언어 시트 단위 에이전트 검수 (읽기 전용, 크레딧 0)
argument-hint: <xlsx 경로> --sheet "JA(일본)" [--semantic-rag-budget N] [--raw]
---

`/st-inspect`는 언어 시트 전체를 읽기 전용으로 검수한다. 원본 Excel은 수정하지 않으며,
리드 에이전트가 공통 근거 패킷을 만든 뒤 5개 관점(문법·의미·현지화·스타일 예외·story/UI 맥락)을
독립 검토하고 보수적으로 취합한다.

- glossary/casing/bracket/brand/navigation path는 앱의 `GlossaryChecker` 결과를 근거 패킷으로
  주입한다. 서브에이전트는 이를 재계산하거나 덮어쓰지 않는다.
- 주관적 수정은 서로 다른 두 관점의 지지와 반대 의견 부재가 있어야 제안한다. 그 외는
  `human_review_queue`로 보낸다.
- offline RAG는 자율적으로 사용한다. semantic RAG는 사용자가 이 시트에 승인한
  `--semantic-rag-budget N` 안에서만 사용하며, 생략하면 0회다.
- v1에는 구조/서식 판정용 결정론적 체커가 없다. 병합·수식·보호·숨김 검증은 Excel 적용 단계의
  안전장치이지 검수 판정 근거가 아니다.

먼저 아래 명령으로 근거 패킷을 만들고, 그 결과를 5개 서브에이전트와 리드 에이전트에 공유한다.

```bash
python agent-packages/smartthings-translation-agent/scripts/agent_sheet_review.py \
  <workbook.xlsx> --sheet "JA(일본)" --semantic-rag-budget 0 \
  --glossary <Glossary.csv> --app-root <app-root> --json
```

리드가 취합한 제안과 검수 context는 `review_report_builder.py`로 v2 리포트와
`pending_approval` manifest로 만든다. 승인된 `changes[]`만 `/st-apply`가 반영한다.

기존 구조·셀 덤프만 필요하면 다음처럼 사용한다.

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_inspect.py \
  <workbook.xlsx> --sheet "JA(일본)" --sections --json
```
