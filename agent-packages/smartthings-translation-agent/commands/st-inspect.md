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

리드는 아래에 해당하면 escalation을 **제안**한다(스스로 켜지 않는다). 비용·지연이 5배로 늘어나므로
사유를 밝히고 승인받는다.

- UI 활성화 조건·면책 문구처럼 검토 축이 교차해 한 관점으로 판정하기 어려울 때
- Pass 2에서 의미 충실도와 현지화 톤의 판단이 서로 충돌할 때
- 같은 오류 유형이 여러 story·시트에서 반복 확인될 때
- 고위험 locale이거나 납품 직전 최종 확인이 필요할 때

**1. 근거 패킷 생성.** 출력에는 `packet_id`가 들어 있고, 모든 의견서가 이 값을 되돌려줘야 한다.
`--multi-agent` 없이 만든 패킷은 `review_mode: lead_2pass`이며 역할 프롬프트 생성과 병합이
거부된다.

```bash
python agent-packages/smartthings-translation-agent/scripts/agent_sheet_review.py \
  <workbook.xlsx> --sheet "JA(일본)" --semantic-rag-budget 0 --multi-agent \
  --glossary <Glossary.csv> --app-root <app-root> \
  --activation-manifest <inactive_manifest.json> --json > packet.json
```

`--activation-manifest`는 용어집 단어가 원문에서 **보통명사로 쓰인 occurrence**를 비활성으로 알려준다.
이게 없으면 `a safe home`의 `safe`가 제품 용어 `Safe`로 계산되어, 맞는 번역이 하드룰 위반으로 차단된다
(ES_CO 실측 5건). 매니페스트를 만드는 절차는 `references/glossary-report-workflow.md`의
"미적용(비활성) 후보 자동 추출"에 있다.

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
  --report-id <id> --output-dir <out> \
  --glossary <Glossary.csv> --app-root <app-root>
```

`--glossary`/`--app-root`는 1단계에서 glossary를 준 경우 **필수다.** 합의를 통과한 제안문을
앱 resolver에 다시 걸어 용어집·대소문자·bracket 정책을 재검증하기 위한 것이고, 검증할 근거가
패킷에 있는데 검증 없이 제안을 내보내는 경로는 막혀 있다.

승인된 `changes[]`만 `/st-apply`가 반영한다.

## 판정 규칙

- 리드 에이전트는 의견서를 요약해 `changes[]`를 만들 수 없다. `review_report_builder`는
  merge 결과 객체만 받으므로 이 경로는 코드에서 막혀 있다.
- **한 역할이라도 의견서가 없거나, `packet_id`가 다르거나, 실행이 오류·중단으로 끝나면 시트 상태는
  `incomplete`다.** 이때는 제안을 하나도 만들지 않고 후보를 전부 `human_review_queue`로 보낸다.
  재시도하지 않는다.
- 검수 시점 셀 값과 현재 값이 다르면 그 제안은 `source_drift`로 보류된다.
- **합의를 통과해도 제안문 자체가 하드룰을 어기면 제안이 되지 않는다.** 병합 직전 모든 `after`가
  앱 resolver에 다시 걸린다. 의견서의 `constraint_status`는 에이전트의 자기신고이므로 신뢰하지
  않고 덮어쓴다. 대소문자·bracket처럼 resolver가 소유한 차이는 **교정**해서 제안에 반영하고
  (`resolver_repaired`), 용어집 target 자체가 어긋나면 `blocked_by_resolver_revalidation`으로
  큐에 보낸다 — 정답으로 조용히 치환하지 않는다. 그 불일치는 사람이 볼 문제다.
- **한 역할의 지지가 다른 역할의 사본에 가까우면 그 지지는 독립으로 세지 않는다.** 남은 독립
  관점이 2개 미만이면 제안이 되지 않고 `anchored_support_needs_independent_role`로 큐에 간다 —
  기각이 아니라 제3의 독립 관점을 요구하는 것이다. 정당한 합의라면 다른 역할이 지지한다.
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
