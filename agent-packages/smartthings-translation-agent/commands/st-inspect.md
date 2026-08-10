---
description: 언어 시트 단위 에이전트 검수 (읽기 전용, 크레딧 0)
argument-hint: <xlsx 경로> --sheet "JA(일본)" [--semantic-rag-budget N] [--raw]
---

`/st-inspect`는 언어 시트 전체를 읽기 전용으로 검수한다. 원본 Excel은 수정하지 않으며,
리드 에이전트가 공통 근거 패킷을 만든 뒤 **리드 에이전트가 직접 배정한 프론티어급 서브에이전트 5명**이
각 관점(문법·의미·현지화·스타일 예외·story/UI 맥락)을 독립 검토하고, 리드 에이전트가 보수적으로 취합한다.

> 실행 주체 구분: API 모델을 역할별로 호출한 결과는 **app/API 검수**다. 이는 `/st-inspect`의
> 서브에이전트 판정으로 표기하거나 이를 대체할 수 없다. API 결과는 필요한 경우 별도 섹션의 보조 근거로만
> 병기한다.

- glossary/casing/bracket/brand/navigation path는 앱의 `GlossaryChecker` 결과를 근거 패킷으로
  주입한다. 서브에이전트는 이를 재계산하거나 덮어쓰지 않는다.
- 주관적 수정은 서로 다른 두 관점의 지지와 반대 의견 부재가 있어야 제안한다. 그 외는
  `human_review_queue`로 보낸다.
- offline RAG는 자율적으로 사용한다. semantic RAG는 사용자가 이 시트에 승인한
  `--semantic-rag-budget N` 안에서만 사용하며, 생략하면 0회다.
- v1에는 구조/서식 판정용 결정론적 체커가 없다. 병합·수식·보호·숨김 검증은 Excel 적용 단계의
  안전장치이지 검수 판정 근거가 아니다.

## 실행 순서

기본 경로는 **리드 에이전트의 시트 전체 2-pass**다. 5개 관점 병렬 검수는 사용자가 그 시트에
`--multi-agent`로 승인했을 때만 켜지는 escalation이며, 아래 2~4단계는 그 경우에만 수행한다.

**1. 근거 패킷 생성.** 출력에는 `packet_id`가 들어 있고, 모든 의견서가 이 값을 되돌려줘야 한다.
`--multi-agent` 없이 만든 패킷은 `review_mode: lead_2pass`이며 역할 프롬프트 생성과 병합이
거부된다.

```bash
python agent-packages/smartthings-translation-agent/scripts/agent_sheet_review.py \
  <workbook.xlsx> --sheet "JA(일본)" --semantic-rag-budget 0 --multi-agent \
  --glossary <Glossary.csv> --app-root <app-root> --json > packet.json
```

**2. 역할별 프롬프트를 만들어 5개 서브에이전트를 병렬 배정한다.** 프롬프트는 즉석에서 쓰지 않고
빌더로 생성한다. 이 빌더는 패킷만 입력으로 받으므로 다른 역할의 의견이 섞일 수 없다.

```bash
python agent-packages/smartthings-translation-agent/scripts/agent_role_prompts.py \
  --role grammar_fluency --packet packet.json
```

5개 호출은 **한 번에 병렬로** 발행한다. 순차 실행하면 지연이 5배가 된다.

**3. 각 서브에이전트의 의견서를 `<dir>/{role}.json`으로 저장한다.** 발견이 없어도
`status: "no_findings"`와 빈 `opinions`로 완료 의견서를 남긴다. 대화로만 말하고 파일을 남기지
않으면 그 역할은 수행되지 않은 것으로 처리된다.

**4. 병합해 v2 리포트와 manifest를 만든다.**

```bash
python agent-packages/smartthings-translation-agent/scripts/agent_sheet_merge.py \
  --packet packet.json --opinions-dir <dir> --workbook <workbook.xlsx> \
  --report-id <id> --output-dir <out>
```

승인된 `changes[]`만 `/st-apply`가 반영한다.

## 판정 규칙

- 리드 에이전트는 의견서를 요약해 `changes[]`를 만들 수 없다. `review_report_builder`는
  merge 결과 객체만 받으므로 이 경로는 코드에서 막혀 있다.
- **한 역할이라도 의견서가 없거나 `packet_id`가 다르면 시트 상태는 `incomplete`다.** 이때는
  제안을 하나도 만들지 않고 후보를 전부 `human_review_queue`로 보낸다. 재시도하지 않는다.
- 검수 시점 셀 값과 현재 값이 다르면 그 제안은 `source_drift`로 보류된다.
- 한 역할의 지지가 전부 다른 역할과 (finding_id, after)까지 같으면 독립 근거가 아닐 수 있으므로
  리포트에 독립성 경고가 붙는다. 자동 차단은 하지 않는다.
- 모든 finding에 `row_type`(title/description/disclaimer/button)이 붙고 리포트에 유형별로
  집계된다. 관점(role) 축과 콘텐츠 유형 축 중 어디로 결함이 뭉치는지 판단할 근거다.

리포트에는 아래 실행 주체를 구분해 기록한다.

- `/st-inspect`: `실행 주체: 프론티어 서브에이전트 5명`
- app 검수: `실행 주체: app API (모델명)`
- RAG: `실행 주체: RAG 조회`
- 최종 판정: `실행 주체: 리드 에이전트`

기존 구조·셀 덤프만 필요하면 다음처럼 사용한다.

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_inspect.py \
  <workbook.xlsx> --sheet "JA(일본)" --sections --json
```
