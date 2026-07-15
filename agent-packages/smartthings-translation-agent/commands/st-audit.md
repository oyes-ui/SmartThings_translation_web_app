---
description: 앱 검수(inspection) 파이프라인 실행 [LLM 크레딧]
argument-hint: <xlsx 경로> --sheets <대상>
---

⚠️ **LLM 크레딧 소모.** 실행 전 먼저 **크레딧 0 self audit**을 권하라:

1. `/st-sections`(`workbook_inspect.py --sections`)로 story 단위 그룹을 뽑고, 기존 RAG 리포트/사례(`rag_lookup.py`, offline 가능)를 곁들여 검수 6대 항목을 **story 단위**로 먼저 훑는다.
2. 이 self audit으로 좁혀진 **후보 셀만** `/st-prompt --audit`(셀프, 크레딧 0)로 개별 검수한다.
3. 대량·자동화·재현 가능한 산출물이 꼭 필요할 때만, 사용자 승인 후 `--pipeline`으로 실행한다.

결과 제시는 `references/response-patterns.md`의 A(검수 등급 설명)·C-2(섹션·story 맥락 검토) 템플릿을 쓴다.

사용자 승인 후에만 `--pipeline` 으로 실행:

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_audit.py $ARGUMENTS --pipeline --json
```
`--pipeline` 없으면 거부+셀프 안내. `.audit_report.txt` 생성. 참조: `agent-packages/smartthings-translation-agent/references/self-vs-pipeline.md`
