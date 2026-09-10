---
description: 번역 검수하기 — 보고서와 수정 제안 (Excel 원본 불변)
argument-hint: <xlsx 경로> --sheet "JA(일본)" [--semantic-rag-budget N] [--raw]
---

# /st-inspect

파일·언어·검수 요청은 자연어로 받는다. 기본은 아래 단계별 에이전트 검수다.
기존 앱 API 검수를 명시적으로 승인한 경우에만 `scripts/workbook_audit.py <xlsx>
--sheets "<대상>" --pipeline --json`을 사용한다. API 크레딧 확인·재시도 규칙은
`references/self-vs-pipeline.md`를 따른다. `/st-audit`도 기존 고급 경로로 유지한다.
검수 결과의 Excel 반영은 `/st-apply`에 연결한다.

기본 구조는 **셀 순차 검수 1명 → 시트 일관성 검수 1명 → 리드 통합 1명**이다. 원본 Excel은
수정하지 않고 resolver를 통과한 제안만 `pending_approval` manifest에 기록한다. 기존 5역할
`--multi-agent` 경로는 deprecated 호환 경로다.

## 1. 근거 패킷

```bash
python scripts/agent_sheet_review.py <workbook.xlsx> --sheet "JA(일본)" \
  --semantic-rag-budget 0 --glossary <Glossary.csv> --app-root <app-root> \
  --activation-manifest <inactive_manifest.json> --json > packet.json
```

확인된 비활성 occurrence가 없으면 `--activation-manifest`를 생략한다. 패킷에는 미확인 비활성
후보가 자동으로 들어가며, 후보와 일치하는 `missing_glossary_target`만
`glossary_activation_review`로 보낸다. 후보 확인 절차는
`references/glossary-report-workflow.md`의 "미적용(비활성) 후보 자동 추출"을 따른다.

## 2. 셀 순차 검수와 게이트

```bash
python scripts/agent_stage_prompts.py --stage cell --packet packet.json > cell_prompt.txt
# 한 에이전트가 cell_prompt를 수행해 cell_review.json 저장
python scripts/agent_stage_gate.py --stage cell_review --packet packet.json \
  --input cell_review.json --output cell_review.validated.json \
  --glossary <Glossary.csv> --app-root <app-root>
```

각 셀은 앞 셀 참조 여부와 참조 셀을 반드시 기록한다.

## 3. 시트 일관성 검수와 게이트

```bash
python scripts/agent_stage_prompts.py --stage sheet --packet packet.json \
  --cell-review cell_review.validated.json > sheet_prompt.txt
# 별도 에이전트가 sheet_prompt를 수행해 sheet_review.json 저장
python scripts/agent_stage_gate.py --stage sheet_consistency_review --packet packet.json \
  --input sheet_review.json --output sheet_review.validated.json \
  --glossary <Glossary.csv> --app-root <app-root>
```

시트 에이전트는 새 일관성 쟁점을 제시할 수 있지만, 영향받는 모든 셀의 전체 수정문과 통일 기준을
남겨야 한다.

## 4. 리드 통합과 리포트

```bash
python scripts/agent_stage_prompts.py --stage lead --packet packet.json \
  --cell-review cell_review.validated.json --sheet-review sheet_review.validated.json > lead_prompt.txt
# 리드가 lead_prompt를 수행해 lead_review.json 저장
python scripts/agent_staged_merge.py --packet packet.json \
  --cell-review cell_review.validated.json --sheet-review sheet_review.validated.json \
  --lead-review lead_review.json --workbook <workbook.xlsx> \
  --report-id <id> --output-dir <out> --glossary <Glossary.csv> --app-root <app-root>
```

리드는 `cell:<cell>` 또는 그 셀을 포함하는 `sheet:<finding_id>` 근거 안에서만 문구를 다듬을 수
있다. 최종안은 저장 직전에 resolver로 다시 검증한다.

## 강제 규칙

- 세 단계 중 하나라도 누락·중단·packet 불일치면 `incomplete`이며 `changes[]`는 비운다.
- 패킷에 constraint card가 있으면 `--glossary`와 `--app-root` 없이 게이트·병합할 수 없다.
- evidence가 없는 셀은 fail-closed다.
- resolver `blocked`는 어떤 에이전트도 뒤집을 수 없다.
- `glossary_activation_review`는 사람에게 보이지만 적용 후보가 아니다.
- 셀 값이 패킷 생성 후 바뀌었으면 `source_drift`로 보류한다.
- Excel 반영은 사람이 승인한 뒤 `/st-apply`에서만 수행한다.

구조만 확인하려면 `scripts/workbook_inspect.py --sections`를 사용한다.

## 여러 언어를 병렬로 준비할 때

```bash
python scripts/agent_staged_batch.py prepare <workbook.xlsx> \
  --sheets "DE(독일)" "FR(프랑스)" "JA(일본)" --work-dir <batch-dir> \
  --glossary <Glossary.csv> --app-root <app-root> --max-concurrency 3
```

출력 manifest의 `ready_for_agent`에 있는 언어별 prompt를 병렬 배정한다. 각 언어 결과는 해당 job
폴더의 `cell_review.json`, `sheet_review.json`, `lead_review.json`에 순서대로 저장하고 매 단계마다
아래 명령을 실행한다.

```bash
python scripts/agent_staged_batch.py advance --manifest <batch-dir>/batch_manifest.json
```

언어 간에는 병렬이지만 한 언어 내부 단계는 순차다. 조정기는 에이전트/LLM을 직접 호출하지 않으며,
semantic RAG 기본 예산도 시트별 0이다.

## 업무 전체를 이어갈 때

`references/workflow-guide.md`와 `references/workbook-batch.md`를 읽는다. 파일별 용어집 적용안을 확정한 뒤 `workbook_batch.py`로 기존 기능을 연결한다. 기본 API는 초벌·Excel 기입·하이라이트만이며 API 검수·역번역은 명시 요청 때만 실행한다. 초벌 뒤 활성 에이전트가 `ready_for_agent`의 셀→시트→리드 prompt를 수행하고 `advance`를 반복해 상세 MD·통합 승인검토표까지 생성한다. 준비 명령만 실행하고 검수가 완료됐다고 보고하지 않는다. 성공 작업은 계속하며 유료 실패는 자동 재시도하지 않는다.
