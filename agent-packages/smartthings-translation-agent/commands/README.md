# 사용자 명령

이 폴더가 명령 안내의 공통 원본이다. 설치된 명령과 호환 alias는 이 문서를 참조하며
검수 단계나 실행 절차를 복제하지 않는다.

| 명령 | 사용자에게 안내할 용도 | 결과 |
|---|---|---|
| `/st-start` | 시작하기·현재 작업 확인 | 연결 상태·다음 할 일 |
| `/st-ask` | 규칙이나 표현 물어보기 | 근거가 포함된 답변 |
| `/st-translate` | 번역하기 | 번역안 또는 번역 워크북 |
| `/st-inspect` | 번역 검수하기 | 검수 보고서·수정 제안 |
| `/st-edit` | 지정한 내용 수정하기 | 변경안 확인 후 수정본(draft) |
| `/st-apply` | 승인한 내용으로 납품본 만들기 | 반영·하이라이트·검증 완료 파일 |

명령 뒤에는 자연어 요청을 받는다. 선택된 파일·언어·검수 결과·대화의 승인 내용을 활용해
에이전트가 내부 JSON과 경로를 준비한다. 사용자에게 manifest나 work_id를 입력하게 하지 않는다.
대상 작업이 여러 개여서 구분할 수 없을 때만 파일/작업을 짧게 확인한다.

## 흐름

- 번역이 필요하면 `/st-translate`, 품질 판단이 필요하면 `/st-inspect`.
- 사용자가 특정 문장을 지정하면 `/st-edit`. 승인 전 preview, 승인 후 수정본을 만든다.
- 검수 수정안 승인 또는 수정본 납품 요청은 `/st-apply`. 업무 납품 조건까지 검증한다.
- “어디까지 됐어?”는 읽기 전용 상태 확인, “이어서 해줘”는 기존 승인 범위의 작업 재개다.
  상태 확인이 실행·재시도·승인을 유발해서는 안 된다. 입력/계약 변경은 재개로 승인하지 않는다.

## 호환 명령

`/st-review` → `/st-inspect`, `/st-help` → `/st-start`.
`/st-pipeline translate` → `/st-translate`의 승인된 API 경로,
`/st-pipeline audit` → `/st-inspect`의 승인된 API 검수 경로.
기존 이름은 바로 삭제하지 않고 목적 명령으로 위임한다.

## 고급/관리 명령

필요할 때만 안내한다: `/st-setup`, `/st-rules`, `/st-prompt`, `/st-glossary`,
`/st-glossary-filter`, `/st-rag`, `/st-ragdb`, `/st-story-review`, `/st-sections`,
`/st-review-apply`, `/st-story-apply`, `/st-highlight`, `/st-textbook`, `/st-audit`,
`/st-audit-explain`, `/st-review-summary`, `/st-notebooklm`, `/st-obsidian-report`.

검수 방식은 `st-inspect.md`, 쓰기·재개는 `references/command-execution.md`,
번역/API 사용 구분은 `references/self-vs-pipeline.md`를 따른다.

무엇부터 해야 할지 모를 때는 [업무 가이드](../references/workflow-guide.md)의 일곱 시작점에서 입력·판단·산출물·다음 단계를 찾는다. 세부 절차의 정본은 그 가이드이며 도움말에 복제하지 않는다.
