# 시트 검수 워크플로 (`/st-inspect`)

시트 단위 검수와 다중 에이전트 의견 계약의 실행 규약. 이 워크플로는 **읽기 전용**이다. Excel을
만들거나 고치지 않으며, 제안은 전부 `pending_approval` 상태로만 남긴다.

관련 문서: 크레딧·모드 판단은 `self-vs-pipeline.md`, Excel 구조·안전 규약은 `excel-workflow.md`,
RAG 조회 규약은 `rag-workflow.md`, 보고 템플릿은 `response-patterns.md`를 따른다.

## 기본 경로 — 리드 에이전트 2-pass

기본값이며 대부분의 검수는 여기서 끝난다.

1. `agent_sheet_review.py`로 근거 패킷 생성(`packet_id`, `cell_snapshot` 포함).
2. Pass 1: 기존 후보(있다면)를 현재 셀 기준으로 재판정.
3. Pass 2: 후보를 근거에서 제외하고 source↔target 전체를 독립 재독해.
4. 결과를 리포트로 정리. 수정이 필요하면 사람 승인 경로로 보낸다.

## Escalation 경로 — 5개 관점 병렬 검수

**사용자가 시트 단위로 승인해야만** 켜진다(`--multi-agent`). 승인되지 않은 패킷으로는
`agent_role_prompts.py`가 프롬프트를 만들지 않고 `agent_sheet_merge.py`도 병합을 거부한다.

리드는 아래에 해당할 때 escalation을 **제안**한다. 스스로 켜지 않는다.

- UI 활성화 조건·면책 문구처럼 검토 축이 교차해 한 관점으로 판정하기 어려운 경우
- Pass 2에서 의미 충실도와 현지화 톤의 판단이 서로 충돌하는 경우
- 같은 오류 유형이 여러 story·시트에서 반복 확인된 경우
- 고위험 locale이거나 납품 직전 최종 확인이 필요한 경우
- 사람 검토 큐가 이미 커서 판정 근거를 더 두껍게 남겨야 하는 경우

비용·지연이 5배로 늘어나므로, 위 사유를 사용자에게 밝히고 승인받는다.

## 서브에이전트 계약

- 프롬프트는 `agent_role_prompts.py`로만 만든다. 즉석 작성 금지.
- 각 역할은 **근거 패킷만** 받는다. 다른 역할의 의견은 어떤 형태로도 전달하지 않는다.
- 5개 호출은 한 번에 병렬 발행한다. 순차 실행은 지연을 5배로 만든다.
- 각 역할은 `{role}.json` 의견서를 반드시 남긴다. 발견이 없으면 `status: "no_findings"`와 빈
  `opinions`. 대화로만 말하고 파일이 없으면 그 역할은 수행되지 않은 것이다.
- 의견서에는 `packet_id`와 실행 메타데이터(`stop_reason`, 가능하면 `run_id`/`model`/시각/
  `confidence`/`rag_evidence_ids`)를 넣는다.

## 판정 규칙 (코드로 강제됨)

- 리드는 의견서를 요약해 `changes[]`를 만들 수 없다. `build_review_artifacts()`는 merge 결과
  객체만 받는다.
- 역할 누락 / `packet_id` 불일치 / 실행 오류 / 비정상 `stop_reason` → 시트 `incomplete`.
  제안을 하나도 만들지 않고 전부 사람 검토 큐로 보낸다. 재시도하지 않는다.
- 주관적 수정은 **서로 독립인** 두 관점의 지지와 반대 부재가 있어야 후보가 된다.
- 한 역할의 지지가 다른 역할의 사본에 가까우면 그 지지는 독립으로 세지 않는다. 이때는 기각이
  아니라 **제3의 독립 관점**을 요구하며 큐로 보낸다(`anchored_support_needs_independent_role`).
- 하드룰 위반은 합의 게이트 없이 해당 체커 근거로 제안한다.
- 검수 시점 셀 값과 현재 값이 다르면 `source_drift`로 보류한다.

## RAG

offline은 자유롭게 조회한다. semantic은 시트별 사전 승인 예산(`--semantic-rag-budget N`,
기본 0) 안에서만 쓰고, 소진되면 offline 근거만 사용하며 남은 쟁점은 사람 검토 큐로 보낸다.
RAG는 규칙·glossary보다 우선하지 않는다.

## 도구

`agent_sheet_review.py` · `agent_role_prompts.py` · `agent_sheet_merge.py` ·
`review_report_builder.py`(라이브러리) · `rag_lookup.py` · `prompt_preview.py` ·
`glossary_manage.py`(조회)

Excel 쓰기·적용·납품은 이 워크플로가 아니다 → `commands/st-apply.md`. 승인/거절 결과를
outcome ledger에 남기는 것도 그 단계에서 수행한다(`review_outcomes.py`).
