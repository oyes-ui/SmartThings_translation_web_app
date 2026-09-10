# 기존 명령의 실행·상태·재개

사용자 명령 목록은 `commands/README.md`가 정본이다. 이 문서는 에이전트의 내부 실행 지침이다.
자연어로 파일·언어·수정안을 받고, 대화의 승인 범위를 내부 요청에 옮긴다. JSON, 계약 해시,
work_id를 사용자에게 작성하게 하지 않는다. 검토되지 않은 제안은 적용하지 않는다.

## 실행 연결

| 요청 | 기존 도구 | 완료 기준 |
|---|---|---|
| 특정 내용 수정 | `workbook_apply_edits.py` | 검증된 draft와 revision |
| 검수 승인안·F/H 감수안 반영 | `workbook_review_apply.py` | 수정본+하이라이트본의 배치 검증 완료 |
| 직접 지정한 수정으로 납품 | `workbook_story_apply.py` | 지정 언어와 KR/US 하이라이트·배치 검증 완료 |
| 번역·검수 API | 기존 translate/audit `--pipeline` | 별도 비용 승인; 로컬 복구 루프 밖 |

위 세 쓰기 도구는 계약 실행기가 기본이다. 기존 인자와 승인 manifest를 유지한다.
`--dry-run`은 계약 preview만 저장하고 Excel·승인 파일은 만들지 않는다. 실제 실행 호출은
이미 사용자 승인이 있는 경계다. 에이전트가 이때 내부 승인 파일을 계약 해시에 결합하며
동일한 내용의 승인을 다시 질문하지 않는다. 검수 report는 `approved` 행만 반영한다.
`pending_approval`은 보류하고, 승인/거절을 끝낸 행만 기존 outcome 형식으로 기록한다.

## 출력과 현재 작업

입력/요청이 같은 호출은 같은 `.st-runs/<work_id>/contract.json`과 journal을 사용한다.
`--output accepted.xlsx`는 파일명과 출력 기준 폴더를 지정한다. 실제 파일은
`<기준 폴더>/verified/<work_id>/accepted.xlsx`와 `accepted_final.xlsx`에 생긴다.
원본이나 기존 배치를 덮어쓰지 않는다. 사용자에게는 반환된 실제 `revised`/`final` 경로를 안내한다.
수정본은 draft, 납품본은 `artifact_status: delivery`이고 `batch.complete.json`의 독립
검증 레코드가 있는 파일이다. 완료 로그나 파일 존재만으로 납품 완료를 선언하지 않는다.

## “어디까지 됐어?”와 “이어서 해줘”

- 상태 질문은 `workbook_run.read_state(work_root)`로 journal을 읽기만 한다.
  `workbook_apply_edits.py <원본> <edits> --run-contract <contract.json> --dry-run --json`도
  상태 읽기 전용 경로다. 새 기본 `--dry-run` 계획 생성과 구분한다.
- 대화에 연결된 작업의 contract/result 경로를 사용한다. 여러 작업 중 어느 것인지
  알 수 없으면 대상 파일만 확인한다. 임의로 가장 최근 작업을 실행하지 않는다.
- 재개 요청은 같은 원본·인자·승인으로 기존 도구를 다시 호출한다. 검증된 staging을
  재검증한 뒤 재사용하고 완료 작업은 기존 결과를 반환한다.
- 미발행 작업에서 원본·용어집·활성화 manifest·규칙/도구 버전이 달라지면 기존 승인으로 재개하지 않는다.
  실패 원인과 달라진 항목을 제시하고 새 계획을 만든다. 기존 실행 폴더를 지워 시도 한도를
  초기화하거나, 통과를 위해 allowed_diffs를 넓히지 않는다. 기본 명령에서 재계획할 때는
  변경된 입력을 명확한 새 버전 파일로 준비해 새 요청으로 시작한다.
- 값 변경 실패는 차단한다. 지원하는 로컬 실패만 한도 내에서 복구한다. 같은 실패의
  반복·총 시도 한도·blocked 상태는 단순 재실행으로 초기화되지 않는다.
- 이 경로는 Gemini/GPT/RAG API를 호출하지 않는다. API 실패는 별도 승인 규칙을 따른다.

## 유지하는 범위

6개 진입 명령, 기존 CLI와 manifest, 하나의 공용 실행기를 사용한다. 새 상태/재개 명령,
전역 작업 DB, 범용 그래프 DSL, daemon, 역할 분할, writer registry는 추가하지 않는다.
독립 검증은 원본 정본의 5축 사실과 승인된 정확한 diff를 비교한다. 납품 계획의 열 삭제는
별도 좌표 투영으로 검증하고, 하이라이트 계획은 승인 범위의 rich text 변화만 허용한다.

메모 상자의 삭제·이동(VML)과 복잡한 수식·차트·validation 의존성의 열 삭제는 별도 구조 계획이 필요하다. 임의의 `.save()`
스크립트 전체를 막는 전역 하네스는 아니다. 과거 일회용 writer와 단독 고급 highlight/
번역 writer까지 이번 배치 보장을 적용했다고 설명하지 않는다.

이미 발행된 배치는 현재 verifier 버전과 분리해 당시 계약·통과 기록·파일 동일성으로
재조정한다. 현재 입력이 바뀌었어도 역사적 산출물을 재생성하지 않는다. 이것은 최신 규칙의
재검증을 뜻하지 않는다. staging과 새 발행은 현재 verifier·입력 검사를 계속 요구한다.
증분 검수 하이라이트도 계약 경유로 전환했고, 반환된 `output`은 버전 배치 안에 있다.
신규 Python save 증가 검사는 전체 pytest와 `scripts/lint_workbook_writers.py`에서 실행한다.
