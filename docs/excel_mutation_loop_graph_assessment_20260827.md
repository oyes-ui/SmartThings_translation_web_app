# Excel 변경 파이프라인 루프·그래프 도입 제안 평가

> **개정 (2026-08-27)**: 초판의 두 주장을 정정했다. **(1) 원인 귀속** — git 이력 확인 결과
> `workbook_mutation_guard.py`와 "새 일회용 도구는 guard를 import한다"는 지침은 커밋
> `87f0d40`(2026-08-26 10:53)에서 사고 **다음 날** 생겼다. 사고 스크립트가 당시 없던
> 지침을 위반했다고 쓴 것은 틀렸다(§8). **(2) 강제 범위** — "guard에 두면 어떤 경로로
> 실행되든 규칙이 유지된다"는 과장이다. guard도 호출하지 않으면 무력하다(§8).
> 아울러 검증 세분화(§7 보완 1), gitignore 제약(§9), 최종 책임 분리와 검증 레코드
> 결합 요건(§10)을 추가했다.

작성일: 2026-08-27
범위: `agent-packages/smartthings-translation-agent`의 `.xlsx` 변경·검증·납품 경로
참고 자료: 외부 제안 "루프 엔지니어링 / 그래프 엔지니어링으로 스킬 감싸기"


## 2026-09-09 후속 코드 리뷰 반영

증분 하이라이트의 검증 전 공개 문제를 확인하고 공용 계약 실행기로 전환했다.
이미 발행된 배치는 당시 계약·완전한 통과 기록·파일 해시로 재조정하며, 현재 verifier/
입력의 유효성 요구는 staging·새 발행에 유지한다. 신규 save 증가를 막는 AST 검사를
전체 pytest에 연결했고 contract 없는 guard 호출에는 DeprecationWarning을 추가했다.

33개 합성 산출물/약 33 MiB 정본 배치에서 inventory 200회, 누적 해시 입력 약 6.45 GiB,
해시 3.352초/전체 4.498초를 측정했다. warm-cache 로컬 1회 측정이며 큰 실배치의 보장값은
아니다. 검증 게이트는 유지하고, 필요하면 배치 경계에서 중복 해시를 줄인다.

최신 검증은 **269 passed, 22 subtests passed**, 기존 회귀 자산 56개 해시는 그대로다.
리뷰의 “하나를 고치면 모든 사용자 경로가 닫힌다”는 결론은 채택하지 않는다. 고급 단독
writer 전면 이전은 남아 있다. 근거와 범위는 패키지
`docs/excel_post_step5_review_20260909.md`에 기록했다.

## 2026-09-09 후속 구현 — 5단계 완료: 기존 명령 연결

사용자 승인에 따라 기본 진입점을 `st-start`(시작·상태), `st-ask`(질의),
`st-translate`(번역), `st-inspect`(검수), `st-edit`(수정본), `st-apply`(납품본)의
6개로 정리했다. `st-review`/`st-help`/`st-pipeline`은 호환 alias이며 자연어로 요청한다.
정본은 패키지 `commands/README.md`, 실행·재개 지침은 `references/command-execution.md`다.

`workbook_apply_edits.py` 기본 경로와 story/review apply를 기존 contract runner에 연결했다.
수정본과 최종 하이라이트본은 독립 5축 검증이 모두 통과한 뒤 버전 배치로 공개한다.
기존 사용자 승인 경계에서 내부 승인 파일을 결합하고 JSON·work_id 입력이나 중복 승인을
사용자에게 요구하지 않는다. 상태 조회는 읽기만 하며 같은 요청은 journal로 재개한다.

과도한 하네스를 피하려고 추가한 실행 코드는 story/review가 공유하는 납품 어댑터 하나다.
기존 preflight·승인 manifest·glossary resolver·rich-text renderer·revision/outcome 형식을
재사용했다. 새 프레임워크·DB·daemon·role·writer registry·상태 명령은 추가하지 않았다.
API 파이프라인은 로컬 복구 루프 밖에 둔다. 납품 경로가 모델/RAG client를 생성하지 않는다.

검증: 패키지 전체 **260 passed, 22 subtests passed**. 기본 CLI 납품, 동일 작업 재개,
preview 이후 glossary drift 차단, pending 항목 제외, KR/US와 대상 언어 하이라이트,
최종 실패 시 수정본까지 미공개, 열 삭제 뒤 치수·병합·범위 밖 서식 보존을 확인했다.
기존 inventory 56개 파일의 SHA-256은 모두 그대로다. 외부 라이브러리 deprecation warning
2건이 있으며, 실제 납품 워크북 생성과 유료 API 호출은 수행하지 않았다.

제약: `--output`은 파일명/기준 폴더이며 실제 산출물은 `verified/<work_id>/`에 위치한다.
메모 상자의 삭제·이동(VML), 수식·차트 등 복잡한 열 삭제 의존성은 별도 계획이 필요하다.
과거 일회용 writer 전면 이전·단독 고급 writer·전역 lint 강제는 이번 완료 범위가 아니다.
이 절이 아래 1~4단계의 opt-in/다음 단계 설명보다 최신이다.

## 2026-09-09 후속 구현 — 최신 계획의 2~4단계

앞선 1단계에 이어 **실행 contract·독립 verifier → 대표 writer 연결 → 상태·이벤트·재개·제한
복구**를 구현했다. 구현 상세와 실행 명령, 지원 범위는
`agent-packages/smartthings-translation-agent/docs/excel_contract_run_20260909.md`를 따른다.
이 절이 아래 과거 구현 현황보다 최신이다. 역사적 33개 writer 전면 이전이나 과거 파일럿 A/B
전체 완료를 뜻하지 않는다.

- `workbook_contract.py`: 3축 정본, 위치·속성별 정확한 allowed_diffs, no_op_files,
  디렉터리 inventory와 해시, 작업별 경로, local writer 버전·복구 한도.
- `workbook_verifier.py`: 원본 정본을 독립 재개방하여 기대 기준을 계산하고, 관측 변경과
  선언 변경의 정확한 일치 여부를 검증한다. 산출물·계약·입력·verifier 버전에 결합된 레코드.
- `workbook_mutation_guard.py`: 기존 save 함수에 contract 옵션을 연결했다. 공용 상세
  5축 snapshot과 열 삭제 audit를 추가했다. 그림/차트 anchor·payload와 유효성 조건까지
  상세 검증하며, 지원하지 못하는 OOXML 객체와 열 삭제 의존성은 변경 전에 거부한다.
- `workbook_run.py`: 승인된 local writer 실행, journal과 state cache, 프로세스 잠금,
  재개 시 독립 재검증, 실패별 최대 시도·동일 실패 한도, 버전별 배치 전체 공개.
- 대표 writer는 추적되는 `workbook_apply_edits.py`다. 기존 preflight와 revision 형식을
  재사용하는 opt-in `--prepare-run` / `--run-contract` 경로를 연결했다.
  기존 기본 명령을 전환하지 않았고 일반 편집 결과는 여전히 draft다.

### 과도한 하네스에 대한 결정

사용자 지적에 따라 JSON 계약·세 모듈·기존 CLI 옵션으로 범위를 제한했다. 새 프레임워크,
DSL, DB, daemon, dashboard, agent role, writer registry는 만들지 않았다. 상태 저장은
중단 후 재개에 필요한 것만 담당하고 번역 판단이나 모델 호출을 하지 않는다. guard 없는
legacy 경로의 호환을 유지하되 그 경로까지 새 보장을 적용했다고 주장하지 않는다.

### 확인한 동작과 남은 단계

합성 회귀 테스트로 정상 실행·원본 불변·누락/미허용 변경·정본 drift·반복 실패 중단·실제
별도 프로세스 강제 종료 후 재개·동시 실행 잠금·배치 일부 실패/공개 중단을 검증했다.
최종 테스트 수와 명령은 위 구현 문서의 종료 기록을 정본으로 삼는다.

다음 5단계는 `/st-edit`·`/st-apply` 전체 진입점 연결과 운영 writer 순차 이전이다.
기존 writer allowlist/lint 강제, 전체 시트 복사 및 복잡한 구조 변경 planner도 후속 범위다.
사용자 실제 워크북으로 납품본을 생성하거나 API 비용을 쓰는 작업은 수행하지 않았다.

## 2026-09-09 업데이트 — 실행 그래프·복구 루프와 1단계 진행

이 절은 현재 코드 재확인과 사용자 승인에 따른 후속 작업이다. 아래 기존 본문의 수치와
착수 체크리스트 테스트 수는 **2026-08-27 당시 기록**이며 현재 수치로 해석하지 않는다.

### 현재 코드와 기존 진단의 차이

- `save_verified_atomic` 비테스트 호출 위치는 4곳이다: regional build, Korean review build,
  Korean review draft, glossary highlight. 호출 위치 증가는 각 writer 전체의 5축 검증이나
  납품 안전성이 완성됐다는 뜻은 아니다.
- 공용 guard에는 아직 contract 인자·검증 레코드 해시 결합·`promote_verified`가 없다.
- `agent_staged_batch.py`에는 cell → sheet → lead 검수 단계와 게이트가 이미 있다.
  신규 그래프는 이를 재사용하되 파일 존재를 넘어 입력·규칙·계약 버전의 유효성을 확인해야 한다.
- `review_outcomes.py`와 `quality_scorecard.py`는 감수 결과와 품질 평가 기반이다.
  실행 이벤트 원장과 감수 결과 원장은 목적이 다르므로 별도로 유지하고 work_id로 연결한다.
- 2026-09-09 변경 전 로컬 기준 테스트는 **190 passed, 16 subtests passed**였다.
  시스템 Python에 pytest가 없어 app repo의 `venv/bin/python`으로 실행했다.
  ignored 테스트도 포함한 수치이며 fresh clone 재현 결과로 주장하지 않는다.

### 그래프와 루프의 구체적인 개발 목표

공통 실행기는 `입력·정본 고정 → 계획 → 승인 → staging → 독립 검증 → 납품`의
입력·출력·전이 조건을 코드로 표현한다. 기존 명령과 Python 도구를 연결하는 작은 실행기로
시작한다. 프레임워크·지식 그래프 도입은 현재 착수 범위가 아니다.

실패 시 무조건 전체 재실행하지 않는다. 실패 유형에 따라 생성 단계 복귀, snapshot·계획
재생성, 사용자 판단 대기를 구분한다. 최대 재시도 횟수, 같은 실패 반복, 개선 없음 등의
종료 조건을 둔다. 검증을 통과하려고 allowed_diffs를 자동 확장하지 않는다. 승인 내용이
바뀌면 승인을 재사용하지 않으며 API 비용이 드는 재시도는 기존 승인 규칙을 따른다.

입력·용어집·구조 매핑 등 의존 항목이 바뀌면 영향을 받는 후속 결과를 무효화하고,
동일한 입력·계약·버전의 검증된 결과만 재사용한다. 실패·감수 결과는 원인 분류 후
최소 합성 입력과 불변 주장으로 남기고, 공용 코드 개선 뒤 별도 평가 사례에도 적용한다.

### 기존 문서의 보장 범위 보완

1. **5축이 모든 속성을 덮는 것은 아니다.** 현재 snapshot의 그림·차트는 개수,
   데이터 유효성은 sqref 범위 중심이다. anchor·유효성 조건 등 지원 객체별 속성 검증은
   후속 contract/verifier 구현 범위로 명시해야 한다.
2. **파일 atomic replace는 배치 트랜잭션이 아니다.** 전 파일 검증 후에도 승격 중
   중단되면 신·구 파일이 섞일 수 있다. 버전별 배치 디렉터리와 완료 manifest를 만들고
   소비자가 완료된 배치만 선택하는 설계가 필요하다. 현재 guard가 이를 보장한다고 보지 않는다.
3. **import lint는 완전한 통제가 아니다.** guard를 import하고 직접 save하는 코드도
   가능하다. 검증 레코드를 요구하는 실제 납품 경로와 함께 적용해야 한다.
4. **재개에는 해시와 버전이 필요하다.** 산출물 존재만으로 완료·재사용을 판정하지 않는다.

### 첫 번째 단계 — 사고 회귀 자산 보존·분리

사용자 요청에 따라 첫 단계는 재사용 불변식 보존으로 한정한다. contract/verifier/run 구현,
legacy writer 전면 이전, 실제 납품 워크북 생성은 다음 단계다.

- `agent-packages/smartthings-translation-agent/docs/excel_regression_inventory_20260909.json`:
  변경 전 `es_co_*.py` 52개와 `test_es_co_*.py` 4개의 상대경로·크기·SHA-256·ignore 여부.
  해시 목록은 식별 증적이지 원본 코드 복구용 백업은 아니다. 기존 파일은 그대로 유지한다.
- `agent-packages/smartthings-translation-agent/tests/test_workbook_regression_invariants.py`:
  ignored writer를 import하지 않는 합성 입력과 8개 회귀 테스트. 공용 guard만 사용한다.
  rich-text run·공백 보존, run 경계를 걸친 편집, no-op 5축·수식·013 보존,
  잘못된 서식 정본 감지, 실제 행 삭제와 마지막 구분 행 보존, 워크북 간 셀 스타일 보존,
  사고 축별 변형 탐지, 검증 실패 시 신규 납품 파일 미생성을 포함한다.
- `agent-packages/smartthings-translation-agent/docs/excel_regression_step1_20260909.md`:
  사고 테스트와 신규 불변식 대응, 범위 한계, 실행·검증 결과.

여기서 no-op 테스트는 공용 snapshot의 보존·탐지 능력을 검증한다. 아직 없는 실행 contract의
정본 선택 정책까지 구현한 것은 아니다. 워크북 간 셀 스타일 테스트 또한 전체 시트·열 dimension
복사 API의 이전 완료를 뜻하지 않는다. 기존 제품명 치환, glossary activation·seed 정책,
국가·story별 업무 테스트는 ignored 원본에 유지한다. 재사용 불변식 테스트 파일은 ignore 대상이
아니며 커밋 가능한 신규 파일로 준비한다. Git staging/commit 자체는 이번 작업에 포함하지 않는다.

## 목적

Excel 변경 스킬을 상태 그래프 실행기(`Contract → Inventory → Snapshot → Plan → Apply
to Staging → Independent Verify → Promote`)와 실패 분류 폐쇄 루프로 감싸자는 외부
제안을 평가한다.

이 문서의 판단은 제안 텍스트가 아니라 `scripts/workbook_mutation_guard.py`,
`references/excel-workflow.md`, `scripts/es_co_global_merge_260825.py`,
`scripts/es_co_regional_build_260825.py`를 직접 확인한 사실을 기반으로 한다.

`docs/agent_architecture_notebooklm_assessment_20260811.md`의 §6 work ledger가
"별도 설계 필요"로 보류된 항목인데, 이번 제안의 `.st-runs/` 상태 파일이 사실상
같은 요구다. 두 문서를 함께 읽어야 한다.

## 핵심 결론

1. **제안이 신설하자는 노드가 사용할 primitive는 대부분 이미 구현돼 있다.** Snapshot·Verify·
   Staging·Promote의 저수준 부품은 `workbook_mutation_guard.py`에 존재하고, 3축 정본 계약은
   `references/excel-workflow.md`에 규범으로 문서화돼 있다. 다만 실행 시점 contract, 상세
   verifier, 상태 전이와 실행 원장은 아직 일반화되지 않았다.
2. **현재 병목은 채택률이다.** 실제로 `.save()`를 호출하는 33개 쓰기 경로 중 guard 기반
   승격을 쓰는 경로가 사실상 없으며, 사고 후 만들어진 guard로 이전되지 않은 legacy writer가
   남아 있다. `openpyxl`을 쓰는 읽기 전용 파일까지 우회자로 세지 않는다.
3. **따라서 그래프의 가치는 오케스트레이션이 아니라 관문(choke point)이다.** 스크립트가
   여전히 `workbook.save()`를 직접 부를 수 있는 한, 상태 실행기를 얹어도 우회는 그대로다.
4. **사고 당시 원인은 하나가 아니라 둘이다.** 서식·구조 계열 사고는 당시 공용 guard와 상세
   검증 기반이 없어서 검증 축이 누락된 것이고, 값 계열 사고는 정본이 실행 계약으로 고정되지
   않은 데서 왔다. 현재의 문제는 사고 후 도입한 guard로 legacy writer 이전이 끝나지 않았다는
   것이다. 전자는 공용 primitive·상세 verifier·writer 이전으로, 후자는 contract 파일로 해결한다.

## 1. 제안 노드 대비 현재 구현 상태

| 제안 노드 | 현재 구현 | 위치 |
|---|---|---|
| `snapshot_inputs` | `semantic_workbook_snapshot` — 3축이 아니라 **5축** (values / layout / styles / rich_text / annotations) | `workbook_mutation_guard.py` |
| `verify_values` 외 4종 | `snapshot_axis_diff` — 축별 독립 fingerprint 비교 | 동일 |
| `apply_mutation` → `promote_outputs` | `save_verified_atomic` — temp 저장 → **재개방 검증** → `os.replace` | 동일 |
| `structure_mismatch` 보정 | `delete_rows_with_manifest` — 물리 행 삭제 + 병합셀 재배치 + row_dimension 재인덱싱 + 삭제분 audit | 동일 |
| `rich_text_loss` 보정 | `clone_rich_value`, `copy_cell_style`, `copy_row_layout` | 동일 |
| `contract_authorities` (3축 정본) | 값 정본 / 구조 정본 / 서식 정본 표로 문서화 | `references/excel-workflow.md:24-33` |
| 체크포인트 재개 | `resumable_statuses` + resume 필터 | `scripts/batch_co_rollout.py`, `tests/test_batch_resume.py` |

`references/excel-workflow.md:80`은 이미 "새 `.xlsx` 수정 스크립트는
`scripts/workbook_mutation_guard.py`를 공용 기반으로 import한다"고 **강제**하고 있다.
guard의 모듈 docstring도 같은 취지다.

즉 노드가 아니라 노드에 필요한 저수준 부품이 있는 상태다. 실질적으로 미구현인 것은
`contract`(실행 계약), 위치·속성 단위 상세 verifier, 일반화된 상태 전이·체크포인트,
`record_outcome`(실행 원장)이다.

## 2. 실제 병목: 채택률

```
scripts/*.py                              124개
openpyxl을 쓰는 스크립트                     72개
그중 workbook_mutation_guard를 import       2개   (workbook_manifest.py,
                                                 es_co_regional_build_260825.py)
guard를 우회하는 스크립트                    70개
그중 실제로 .save()를 호출하는 것            33개   ← 이전 대상은 이 33개
es_co_* 일회용 스크립트                      52개
```

우회 70개 중 37개는 읽기 전용 분석 도구다. **실제 쓰기 경로는 33개**이며 이전 대상은
이 33개다. `save_verified_atomic`의 프로덕션 호출부는 현재 `es_co_regional_build_260825.py:417`
**1곳**뿐이다(그 외 테스트 2곳).

사고 시점에는 현재의 guard와 일회용 writer 지침이 존재하지 않았다. 이후 guard가 해당 실패
유형을 해결하도록 추가됐지만, 기존 writer 이전과 실행 계약 도입이 아직 끝나지 않았다.

## 3. 260825 사고 스크립트 정밀 분석

`scripts/es_co_global_merge_260825.py`(22KB)는 guard를 import하지 않고
`clone_rich`, `rich_signature`, `copy_cell`, `copy_sheet`, `values_snapshot`,
`structure_snapshot`, `atomic_json`을 직접 재구현했다.

**공정하게 짚을 점**: 복사 로직 자체는 부실하지 않다. `copy_sheet`(:154-183)는
row/column dimension, 병합, `sheet_properties`(탭 색 포함), page setup까지 복사하며
일부는 guard의 `copy_row_layout`보다 넓다. 문제는 복사가 아니라 **검증과 승격 순서**다.

### 결함 1 — 검증이 값·rich-text 2축만 덮는다

`verify_global_output`(:270-298)은 `values_snapshot`만 사용해 manifest 허용 변경과
대조하고 보호 대상 rich-text 변경을 차단한다. 이 부분은 제안이 말하는 `allowed_diffs`
개념을 이미 구현한 좋은 코드다.

그러나 `structure_snapshot`(:198-212)은 **정의만 되어 있고 어디서도 호출되지 않는다.**
게다가 그 함수가 담는 것은 sheetnames / state / merged / freeze_panes / max_row /
max_column뿐이다. guard의 5축 대비 다음이 검증되지 않는다.

| 검증 항목 | guard 축 | 사고 스크립트 |
|---|---|---|
| 셀 서식(font/fill/border/alignment) | `styles_sha256` | 없음 |
| 행 높이 | `row_dimensions` | 없음 (dead code에도 미포함) |
| 열 너비 | `column_dimensions` | 없음 |
| 시트 탭 색 | `tab_color` | 없음 |
| 데이터 유효성 / 이미지 / 차트 | 각 항목 | 없음 |
| 셀 주석 | `annotations_sha256` | 없음 |
| 값 | `values_sha256` | 있음 |
| rich-text run | `rich_text_sha256` | 있음 (`values_snapshot` 내부) |

보고된 사고 중 **탭 색 불일치, 행 5 높이, 서식 불일치**는 정확히 이 미검증 축과 1:1로 대응한다.

### 결함 2 — 검증보다 승격이 먼저 일어난다

```python
# es_co_global_merge_260825.py:318-325
staged = output.with_suffix(output.suffix + ".tmp")
global_book.save(staged)
os.replace(staged, output)          # ← 최종 경로로 승격
...
verification = verify_global_output(global_source, output, changes)   # ← 그 다음 검증
```

`build_co_only`(:347-350)도 동일하다. `save → os.replace → verify` 순서다.

guard의 `save_verified_atomic`은 정확히 반대다.

```python
workbook.save(temporary)
result = verify_path(temporary)     # 임시 파일을 먼저 검증
os.replace(temporary, destination)  # 통과해야 승격
```

따라서 사고 스크립트에서는 검증이 예외를 던져도 **불량 파일이 이미 최종 출력 경로에 놓인
상태**다. 현재 guard는 단일 파일의 검증 후 교체를 제공한다. 전체 배치 통과 게이트나 배치
승격의 원자성까지 구현한 것은 아니다. 또한 guard는 사고 다음 날 추가됐으므로 사고 당시
이미 존재하던 guard를 우회했다고 해석해서는 안 된다.

### 결함 3 — guard에 실제로 없는 기능이 하나 있다

`copy_co_sheet`(:226)와 `build_co_only`(:341)는 `delete_cols(4, 5)`를 호출한다.
guard에는 `delete_rows_with_manifest`만 있고 **열 삭제의 안전 등가물이 없다.**
행 삭제와 같은 위험(병합 셀 경계 교차, dimension 재인덱싱)을 갖는데 audit도 없다.

이것은 "guard를 그래프로 감싸라"가 아니라 "guard를 확장하라"는 신호다.

## 4. 사고 원인의 이중 구조

| 사고 | 원인 유형 | 해결 수단 |
|---|---|---|
| 탭 색 처리 | 검증 축 누락 | guard 독점화 (5축 snapshot 강제) |
| 행 5 / 행 6 혼동, 행 높이 | 검증 축 누락 | 동일 |
| 서식 불일치 | 검증 축 누락 | 동일 |
| 삭제 섹션을 비우기만 함 | 검증 축 누락 + API 미사용 | `delete_rows_with_manifest` 강제 |
| 띄어쓰기 손실 | **정본 미명시** | contract 파일 |
| `013` 값 누락 | **정본 미명시** | contract 파일 |
| KR 051 no-op 파일 변형 | **정본 미명시** | contract 파일 (`no_op_files`) |

값 계열 사고가 특히 중요하다. `verify_global_output`은 값을 실제로 검증하는데도
띄어쓰기·`013`이 새어 나갔다. 이는 탐지기의 문제가 아니라 **어떤 파일을 기준선으로
비교할지가 실행 시점에 고정되지 않았기 때문**이다. 제안의 3축 정본 계약이 필요한 지점은
정확히 여기이며, 여기서는 제안이 옳다.

## 5. 제안 채택 판단

### 채택

- **`contract.json` (3축 정본 + `allowed_diffs` + `no_op_files`)** — 값 계열 사고의 직접적
  해결책. 지금은 규범 문서(`excel-workflow.md`)에만 있고 실행 시점 아티팩트가 없다.
- **`events.jsonl` 실행 원장** — 2026-08-11 평가의 §6 work ledger 재등장. 비용이 낮고
  "이 파일이 왜 이렇게 생성됐는가"의 재구성을 가능하게 한다.
- **사고 → 회귀 fixture 전환** — 가장 싸고, 이후 어떤 재설계에도 살아남는 유일한 자산.
- **실패 분류 → 결정 매핑** — 다만 서브시스템이 아니라 `snapshot_axis_diff` 결과를
  contract의 `allowed_diffs`와 대조하는 함수 수준으로 충분하다.

### 보류

- **YAML 선언형 DSL (제안 4절).** 일회용 스크립트가 52개라는 사실은 작업 형태가 매번
  다르다는 뜻이다. DSL이 표현하지 못하는 케이스가 나오면 사람은 우회로를 만들고, 그것이
  정확히 이번 사고의 메커니즘이다. Python을 유지하되 guard를 유일 통로로 만드는 편이 안전하다.
  작업 형태가 3~4개로 수렴한 뒤에 재검토한다.
- **LangGraph 등 프레임워크 도입.** 제안 본문도 같은 결론이며 동의한다.
- **지식 그래프 (제안 5절).** 현재 질의 수요가 없다. YAGNI.

### 재배치

- **상태 그래프 실행기.** 제안은 이것을 1순위로 두지만, 우회가 열려 있는 상태에서는
  장식이다. 공용 guard·상세 verifier·lint로 표준 경로를 먼저 만든 뒤로 옮긴다.

## 6. 권장 구현 순서

제안의 1→2→3→4→5 대신 다음 순서를 권장한다. 세부 착수 순서는 문서 끝 체크리스트를
정본으로 삼는다.

1. **ignored 회귀 자산 보존·분리.** 사고 스크립트와 테스트의 목록·해시를 기록하고,
   최소 합성 입력과 재사용 불변식을 추적되는 테스트로 옮긴다.
2. **책임 경계와 contract 정의.** guard / contract / verifier / run을 분리하고,
   3축 정본·위치/속성 단위 허용 diff·작업별 경로를 계약으로 만든다.
3. **공용 primitive와 상세 verifier 완성.** `delete_cols_with_manifest`를 추가하고 5축 빠른
   감지 뒤 위치·속성 단위 대조를 수행한다.
4. **검증된 승격 경로.** staged artifact·contract·입력 정본 해시가 결합된 검증 레코드가
   있을 때만 승격한다.
5. **신규 우회 증가 차단.** 실제 `.save()` writer 33개를 allowlist로 동결하고 신규 writer
   lint를 적용한다. 읽기 전용 `openpyxl` 사용 파일은 대상이 아니다.
6. **정상/결함 파일럿 후 공통 실행기 연결.** 동등성 파일럿과 결함 수정 파일럿을 분리하고,
   검증 후 `/st-edit`·`/st-apply`가 공통 실행기를 호출하게 한다.
7. 이후 필요해지면 상태 그래프 실행기 / 지식 그래프를 검토한다.

`.st-runs/` 저장 위치는 신규 최상위 디렉터리를 만들기 전에 2026-08-11 평가의 §6
(`outputs/work/<work_id>/`)과 기존 report·manifest 계약의 관계를 먼저 정리해야 한다.
루트 `CLAUDE.md`의 원자적 쓰기 규약(`.tmp` → `os.replace`)을 따른다.

## 7. 이전 계획(호환성 유지 마이그레이션) 검증

후속으로 제시된 5단계 마이그레이션 계획(현재 상태 고정 → 공용 기능 추가 → 대표 경로 시험
→ 신규 코드부터 강제 → 위험도 순 이전)을 코드 대조로 검증했다. **전체 전략은 타당하다.**
추가만 하고 병렬 비교로 확인한 뒤 신규부터 강제하는 순서는 옳다. 다만 아래는 구현 전에
정정해야 한다.

### 확인된 사실 (계획대로 진행 가능)

- **"기존 33개 저장 스크립트"는 정확하다.** `.save()`를 호출하면서 guard를 쓰지 않는
  스크립트가 정확히 33개다.
- **베이스라인 테스트는 green이다.** `python3 -m pytest tests/ -q` → **184 passed,
  16 subtests, 4.96초.** 실행이 5초라 계획 1단계(현재 상태 고정)는 사실상 비용이 없다.
- **`delete_cols_with_manifest` 신설 필요**는 §3 결함 3과 일치한다.

### 정정 1 — `save_verified_atomic` 호환성 우려는 과대평가다

계획은 시그니처 변경이 기존 호출부를 깰 것을 우려해 `save_contract_verified_atomic`을
별도 신설하자고 한다. 그러나 실제 호출부는 다음이 전부다.

```
scripts/es_co_regional_build_260825.py:417   ← 프로덕션 1곳
tests/test_workbook_mutation_guard.py:114,126 ← 테스트 2곳
```

**함수를 두 개로 늘리는 것은 이 사고의 원인인 "두 개의 문" 구조를 재생산한다.**
안전한 문과 덜 안전한 문이 공존하면 사람은 덜 안전한 쪽을 고른다는 것이 이미 확인된
실패 모드다(70/72).

권장: 별도 함수를 만들지 말고 `save_verified_atomic`에 `contract` 파라미터를 직접
추가한다. 과도기에는 `contract=None`을 허용하되 `DeprecationWarning`을 발생시키고,
33개 이전이 끝나면 필수로 승격한다. 호출부 3곳 수정은 당일 작업이다.

### 정정 2 — `outputs/work/<work_id>/` 구조와 파일명은 실재하지 않는다

계획의 디렉터리 스케치에 있는 `approval_manifest.json` / `delivery_manifest.json`은
현재 규약이 아니다. 실제 `outputs/`는 **평평한 구조**(json 22, xlsx 12, txt 6, md 5)이며
manifest는 산출물의 형제 파일로 놓인다.

```
052_Bixby_다국어_법무검토_반영_260730.manifest.json
052_Bixby_다국어_법무검토_반영_260730.approved.manifest.json
052_다국어_법무반영_최종_260730.delivery.json
```

즉 `<산출물명>.manifest.json`, `<산출물명>.approved.manifest.json`,
`<산출물명>.delivery.json` 패턴이다. **이들을 per-work 디렉터리로 옮기면 기존 소비 코드가
깨지므로 계획 자신의 호환성 목표와 모순된다.**

권장: 기존 manifest는 현재 위치·이름 그대로 둔다. 신규 원장(`contract.json`,
`events.jsonl`, `state.json`)만 새 디렉터리에 두고, manifest는 경로로 **참조**만 한다.
`outputs/work/`도 `.st-runs/`도 아직 없으므로 위치는 구현 시점에 결정하되,
2026-08-11 평가 §6과 정합을 먼저 맞춘다.

참고: `_workspace/2026-08-20-001/`이 존재하지만 `scripts`·`references`·`SKILL.md`·
`commands` 어디에서도 참조되지 않는다. **문서화된 규약이 아니므로 근거로 삼지 않는다.**

### 정정 3 — 3단계 병렬 비교의 기준선이 틀렸다 (가장 중요)

계획은 "기존 결과와 신규 결과를 비교해 계약에 없는 차이가 나오면 신규 경로를 채택하지
않는다"고 하며, 시험 대상으로 `es_co_global_merge_260825.py`를 든다.

**그 스크립트의 기존 산출물은 이미 틀렸다.** 탭 색·행 높이·셀 서식·주석이 검증되지 않은
채 생성됐고(§3 결함 1), 검증 실패 시에도 최종 경로에 승격됐다(결함 2). 기존 출력과의
무차이를 채택 조건으로 걸면 **사고를 사양으로 고정**한다.

기준선은 기존 출력이 아니라 **입력 + contract에서 재계산한 기댓값**이어야 한다.
따라서 파일럿을 두 종류로 분리한다.

| 파일럿 | 대상 | 성공 조건 |
|---|---|---|
| A. 동등성 검증 | `es_co_regional_build_260825.py` (이미 guard 사용, 테스트 보유) | 기존 출력과 **차이 없음** |
| B. 결함 수정 검증 | `es_co_global_merge_260825.py` (사고 스크립트) | 기존 출력과 **차이가 나야 정상**. 차이가 §3의 미검증 축에 국한되는지 확인 |

계획은 이 둘을 "적합한 대상은 A 또는 B"라고 병렬로 제시하는데, 성공 조건이 정반대이므로
섞으면 안 된다.

### 보완 1 — 검증기의 5축 커버리지를 명시할 것

계획의 호환성 판정 기준은 산문으로만 되어 있다. 구현 계약으로 다음을 못박는다.

> `verify_path`는 `semantic_workbook_snapshot`의 5축(`values` / `layout` / `styles` /
> `rich_text` / `annotations`)을 모두 계산한다.

축을 부분적으로만 덮는 자체 스냅샷은 금지한다. 사고 스크립트의 `structure_snapshot`이
정확히 그 사례이며, 그마저도 호출되지 않았다.

**단, 축 단위 허용만으로는 부족하다.** `snapshot_axis_diff`는 "어느 축이 변했는가"만
알려준다. contract에 `"allowed_diffs": ["layout"]`이라고 적으면 행 5 높이 변경을
허용하려던 것이 KR 시트 행 높이 변화·다른 시트 탭 색 변화·열 너비 변화까지 함께
통과시킨다. 사고에서 실제로 문제가 된 것이 정확히 이런 종류의 변화다.

따라서 검증은 2단계여야 한다.

```
1단계  snapshot_axis_diff    → 어느 축이 변했는가 (빠른 이상 감지)
2단계  상세 verifier          → 그 축 안에서 어느 시트·행·셀·속성이 변했는가
                              → 그 변화가 contract에 적힌 것과 정확히 일치하는가
```

`allowed_diffs`는 축 목록이 아니라 위치·속성 단위 선언이어야 한다.

```json
{
  "allowed_diffs": {
    "layout": [
      {"sheet": "CO(콜롬비아)", "property": "row_height", "row": 5,
       "before": 15, "after": 17},
      {"sheet": "CO(콜롬비아)", "property": "tab_color", "after": "FFFF00"}
    ]
  }
}
```

**참고 구현이 이미 있다.** 사고 스크립트의 `verify_global_output`(:284-291)은 값 축에
대해 정확히 이 방식을 쓴다 — 셀 단위 `allowed` 집합을 만들고 `actual != allowed`로
대조한 뒤 `unexpected` / `missing`을 분리해 보고한다. **값 축의 이 패턴을 layout·styles·
rich_text·annotations 축으로 확장하는 것이 상세 verifier의 정확한 정의다.** 사고 스크립트는
값 축에서는 옳았고, 나머지 네 축에서 이 검증이 아예 없었다.

### 보완 2 — 사고 스크립트의 개별 결함을 이전 항목으로 명시할 것

계획에는 판정 기준으로 간접 언급만 있다. 다음 두 건은 명시적 수정 항목이어야 한다.

- `structure_snapshot`(:198-212) 제거 또는 실제 호출 — 현재 dead code
- `build_one`(:318-325) / `build_co_only`(:347-350)의 `save → os.replace → verify`를
  `save → verify → os.replace`로 역전

### 보완 3 — 회귀 fixture의 입력 확보를 1단계에서 먼저 확인할 것

계획 1단계는 "콜롬비아 사고 7건을 회귀 fixture로 작성"한다고만 되어 있다. 저장 형태를
명시해야 한다.

**해소됨 (§9 참조).** 초판은 사고 당시 입력 워크북 확보를 리스크로 봤으나, 확인 결과
`tests/`에 바이너리 `.xlsx` fixture가 0개이며 모든 입력을 코드로 합성한다. 실제 사고 파일은
필요 없고, 남길 것은 **최소 합성 입력 빌더 + 불변 주장** 둘뿐이다. 이미 확립된 방식이므로
추가 작업량이 아니다.

### 보완 4 — 4단계 강제의 구체 형태

"신규 코드부터 강제"는 다음 형태를 권장한다.

- 현재 33개를 명시적 allowlist로 **동결**한다 (파일명 하드코딩).
- lint 테스트: `openpyxl`을 쓰고 `.save()`를 호출하면서 guard를 쓰지 않는데 allowlist에
  없는 파일 → 실패.
- allowlist는 **추가만 금지**하고 제거는 자유롭게 한다. 이전이 진행되면 자연히 줄어든다.

이렇게 하면 현재 업무를 멈추지 않으면서 우회 경로 증가만 차단된다.

## 8. 통합 CLI(`st_excel.py`) 제안 검증

세 번째로 제시된 "기존 Excel 도구 앞에 계약·상태·검증·승격을 담당하는 얇은 통합 관제 CLI를
둔다"는 안을 코드 대조로 검증했다. **결론: 사용성 개선으로는 타당하나, 강제 수단으로는
작동하지 않는다. 그리고 이 통합은 이미 두 번 수행됐고 사고를 막지 못했다.**

### 확인된 사실

- "argparse를 가진 작은 CLI가 많다"는 정확하다. **124개 중 103개**가 argparse를 가진다.
- 어댑터 방식은 실현 가능하다. `es_co_global_merge_260825.py`의 `run()`,
  `run_supplemental()`, `build_one()`은 import 가능한 함수로 분리돼 있다.
- 오버엔지니어링 경계 목록(DSL, LangGraph, DB, 대시보드, 52개 전면 재작성)은
  §5의 보류 판단과 일치한다.

### 반증 1 — "흩어진 도구를 하나의 입구로 묶는다"는 이미 완료됐다

이 패키지에는 **`commands/`에 28개의 slash command가 이미 존재**하며,
`commands/README.md`는 Primary / Legacy 계층까지 명시하고 있다. 더 결정적으로
`commands/st-apply.md`의 첫 문장은 다음과 같다.

> `/st-apply`는 최종 납품 경로다. 기존 `st-review-apply`, `st-story-apply`,
> `st-highlight`의 검증 책임을 통합한다.

이것이 정확히 CLI 제안이 하려는 일이다. **여러 도구의 승격·검증 책임을 단일 입구로 통합하는
작업은 이미 수행됐다.**

### 반증 2 — 그런데도 사고가 났다. 사고 스크립트는 어떤 입구도 통과하지 않았다

`es_co_global_merge_260825.py`와 `es_co_regional_build_260825.py`는
`commands/`·`references/`·`SKILL.md` **어디에도 등록되어 있지 않다**(grep 결과 0건).
ad-hoc으로 직접 실행됐다.

즉 사고의 원인은 "입구가 여러 개라 헷갈렸다"가 아니라 **"입구를 통과하지 않았다"**이다.
네 번째 입구(`st_excel.py`)를 추가해도 등록되지 않은 스크립트는 그대로 우회한다.
이는 `/st-apply` 통합이 이미 겪은 결과의 반복이다.

### 귀속 정정 — "진입점을 안 썼다"가 아니라 "진입점이 없는 작업이었다"

위 반증 2를 "사용자가 slash command를 확인하지 않았다" 또는 "진입점 사용을 강제하자"로
읽으면 안 된다. 두 가지가 이를 배제한다.

**첫째, 해당 작업을 커버하는 명령이 존재하지 않았다.** `/st-edit`의 입력 계약은 셀 단위
before/after다.

```json
[{"sheet":"JA(일본)","cell":"C10","before":"기존 문구","after":"승인된 문구"}]
```

260825 작업은 CO 시트 삽입, `delete_cols(4, 5)`, 시트 순서 이동, 제품명 일괄 치환,
워크북 간 병합이다. 이 계약으로 표현할 수 없다. `/st-apply`는 승인 manifest 기반 납품본
경로이므로 마찬가지다. **새 스크립트를 작성하는 것 외의 선택지가 없었다.**

**둘째, 일회용 스크립트는 우회가 아니라 설계된 경로다.**
`references/excel-workflow.md`에는 "새 일회용 도구의 코드 기반" 절이 존재하고,
`workbook_mutation_guard.py`의 docstring도 "Canonical safety primitives for SmartThings
**one-off** Excel mutation tools"이다. 시스템은 일회용 스크립트의 생성을 전제하고
설계됐다.

**셋째, 사고 시점에는 guard도 그 지침도 존재하지 않았다.** git 이력으로 확인한 시각은
다음과 같다.

| 시각 | 사건 |
|---|---|
| 2026-08-25 21:22 | `es_co_global_merge_260825.py` 최종 수정 (사고 스크립트) |
| 2026-08-26 10:26 | `es_co_regional_build_260825.py` 작성 — guard 사용 |
| 2026-08-26 10:30 | `workbook_mutation_guard.py` 작성 |
| 2026-08-26 10:53 | 커밋 `87f0d40` "Harden workbook mutation and manifest validation" — guard 292줄 + `excel-workflow.md` 73줄(일회용 도구 지침 포함) + 테스트 135줄 |

즉 **guard와 "새 일회용 도구는 guard를 import한다"는 지침은 사고 다음 날 아침에,
사고에 대한 대응으로 만들어졌다.** 사고 스크립트가 존재하지 않던 규칙을 위반했다고
말할 수 없다.

따라서 귀속은 다음과 같다.

| 후보 | 판정 |
|---|---|
| 사용자가 slash command를 확인하지 않음 | **아님** — 해당 작업용 명령이 없었다 |
| 작업에 맞는 기존 명령이 없었음 | **맞음** |
| 사고 당시 일회용 도구의 공용 안전 기반이 없었음 | **맞음 — 직접 원인** |
| 사고 당시 검증·승격 규칙이 코드로 강제되지 않았음 | **맞음 — 직접 원인** |
| 사고 후 추가된 현재 지침을 과거 작성자가 위반함 | **아님** — 지침이 13시간 뒤에 생겼다 |
| 지금 같은 방식으로 새 스크립트를 만들면 지침 위반 | **맞음** — 이후부터 적용 |

정확한 서술은 다음이다.

> 당시 구조 변경 작업을 처리할 공용 기능과 실행 계약이 없었고, 일회용 도구가 자체
> 구현되면서 검증 축이 빠졌다.

이 차이는 설계에 직결된다. 사람의 불이행 문제로 보면 lint만 추가하게 되고, **시스템 기능
부재 문제로 보면 `delete_cols_with_manifest`·contract·상세 verifier를 제대로 만들게 된다.**
후자가 옳다.

### 남은 작업의 성격 — 진행 중인 대응의 완결

위 이력이 보여주는 더 정확한 그림은 다음이다. `87f0d40`은 이미 사고 대응이었고,
`es_co_regional_build_260825.py`는 그 대응으로 guard 위에 다시 쓰였다. **그러나
`es_co_global_merge_260825.py`는 이전되지 않은 채 남았다.**

따라서 이 문서가 제안하는 작업은 "과거 위반에 대한 재발 방지"가 아니라 **2026-08-26에
시작된 대응의 미완 부분을 마무리하는 것**이다. 우선순위가 달라진다.

1. `delete_cols_with_manifest` — 대응 당시 빠진 기능 (§3 결함 3)
2. `es_co_global_merge_260825.py` 이전 — 대응 당시 남겨진 스크립트
3. contract·상세 verifier — 대응 당시 다루지 않은 층위

### 모순 — `promote` 독점은 CLI로 구현할 수 없다

제안은 CLI의 최대 가치로 승격 권한 독점을 든다.

> 일회용 스크립트는 얼마든지 새로 만들 수 있습니다. 하지만 그 스크립트가 바로 최종 폴더를
> 덮어쓸 수는 없게 합니다.

그러나 같은 문서의 "CLI가 해결하지 못하는 것"에는 이렇게 적혀 있다.

> 기존 스크립트를 직접 실행해 CLI를 우회하는 경우

**두 진술은 양립할 수 없다.** Python 스크립트는 언제든 `os.replace()`로 임의 경로에 쓸 수
있다. CLI는 관례이지 기전이 아니다. 사고 스크립트가 `.save()` → `os.replace()`를 직접
호출한 것이(§3 결함 2) 바로 그 증거다.

### 올바른 위치 — 강제는 guard에, 사용성은 CLI에

승격 독점을 **실제로** 성립시키려면 라이브러리 불변식이어야 한다.

```
save_verified_atomic(...)   → contract가 선언한 staging root 밖의 destination을 거부
promote_verified(...)       → contract가 선언한 delivery root로 쓰는 표준 함수.
                              통과한 verification 레코드를 인자로 요구
```

staging/delivery root를 프로그램에 하나로 하드코딩하면 사용자가 지정하는 외부 작업 폴더와
호환되지 않는다. 두 경로는 매 작업 contract에 명시하고 guard가 그 선언 범위 안인지 검사한다.

**단, guard도 절대적 강제 장치는 아니다.** guard를 호출하지 않고 `workbook.save()`를
직접 부르면 guard 역시 아무것도 막지 못한다. 강제 수단을 정직하게 정리하면 다음과 같다.

| 계층 | 막는 것 | 우회 방법 |
|---|---|---|
| CLI | 없음 | 스크립트 직접 실행 |
| guard 불변식 | guard를 호출한 코드의 잘못된 승격 | guard를 호출하지 않음 |
| lint / 테스트 | 저장소 안의 우회 코드 | 저장소 밖 코드 (→ §9) |
| OS 권한 분리 | 실제 쓰기 | (실질적 우회 없음) |

따라서 "guard에 두면 우회할 수 없다"가 아니라 **"저장소 수준의 구조적 강제"**라고 부르는
것이 정확하다. 로컬 개인 작업 환경에서 delivery 폴더 쓰기 권한을 별도 프로세스로 분리하는
것은 현 단계에서 과하다. 현실적 조합은 다음이다.

```
공용 guard + 신규 writer lint + 기존 writer allowlist 동결 + 회귀 테스트 + 납품 경로 검증
```

CLI는 이 불변식 위의 UX 계층이지 그 대체물이 아니다. 제안이 그린 흐름

```
스크립트 → staging → st-excel verify → st-excel promote → 최종 폴더
```

에서 저장소 수준의 보호를 만드는 핵심은 CLI 두 칸이 아니라 **`promote_verified`가 현재
staged artifact와 결합된 검증 레코드를 요구한다는 사실**이다. 단, 직접 저장을 탐지하는 lint와
회귀 테스트가 함께 있어야 이 경로의 우회를 발견할 수 있다.

### 계층 수 문제

`st_excel.py`를 넣으면 실행 경로가 세 계층이 된다.

```
slash command (/st-apply, /st-edit)  →  st_excel.py  →  개별 스크립트
```

`/st-apply`가 이미 납품 관제 역할을, `/st-edit`이 일반 수정 관제 역할을 맡고 있으므로
중간 계층의 책임이 겹친다.

**다만 slash command와 CLI는 같은 종류의 것이 아니다.** slash command는 에이전트가 읽고
따르는 **산문 지침**이고, CLI는 실제로 실행되는 **결정론적 프로그램**이다.
`/st-apply.md`가 "최종 납품 경로다"라고 선언해도 그것이 contract를 기계적으로 검사하거나
상태를 기록한다는 뜻은 아니다.

이 구분이 오히려 반증 2를 강화한다. `/st-apply`의 통합이 사고를 막지 못한 이유는 그 통합이
**산문 계층에만 있었기 때문**이다. 따라서 올바른 답은 "네 번째 산문 입구"도 "아무것도
만들지 않음"도 아니라 **기존 입구들이 공통으로 호출하는 결정론적 실행기**다.

```
/st-edit,  /st-apply          ← 산문 지침 (유지)
        ↓
   공통 Excel 실행기            ← 신설. 새 최상위 UX가 아니라 공통 엔진
        ↓
contract → staging → 상세 verify → promote
```

즉 `st_excel.py`를 사용자에게 새 명령으로 노출할 필요는 없지만, 기존 slash command들이
공통으로 호출하는 실행기로 만드는 가치는 있다. 이 형태에서 CLI는 네 번째 입구가 아니라
기존 입구들의 공통 엔진이다.

### 판정

| 항목 | 판정 |
|---|---|
| 일관된 JSON 출력·종료 코드 | 채택 — 실질적 개선 |
| 기존 스크립트 어댑터(`OPERATIONS` 레지스트리) | 채택 — 실현 가능 확인 |
| `contract` / `status` 하위 명령 | 채택 |
| `promote`를 CLI 전용 권한으로 두는 것 | **불가** — guard 불변식으로 구현 |
| 새 최상위 **사용자** CLI 신설 | **보류** — 사용자에게 새 명령을 노출할 필요 없음 |
| `/st-edit`·`/st-apply`가 공통 호출하는 **내부 실행기** | **채택** — 산문 통합과 달리 기계적으로 구속력이 있다 |
| `outputs/work/<work_id>/` 구조 | §7 정정 2 참조 (실재하지 않음, manifest 이동 금지) |

### 순서

실행기를 먼저 만들면 `/st-apply` 통합이 겪은 결과를 반복한다. 순서는 다음이어야 한다.

1. guard 불변식(staging/delivery 경로 분리 + `promote_verified`) + 상세 verifier
2. 우회 탐지 lint (§7 보완 4, §9 제약 확인)
3. 그다음 공통 실행기 — 표준 경로는 guard가 보호하고, 저장소 안의 우회는 lint가 탐지하는
   상태에서 순수한 사용성 개선으로 추가한다

## 9. 신규 발견 — 일회용 스크립트와 그 테스트가 git에서 제외돼 있다

검증 중 확인한 사실이며, 앞선 세 제안 어디에도 반영되지 않았다. **lint·allowlist·회귀
테스트 계획에 직접 영향을 준다.**

`.gitignore:55-66`은 다음을 제외한다.

```
agent-packages/smartthings-translation-agent/scripts/es_co_*.py
agent-packages/smartthings-translation-agent/scripts/story_*.py
agent-packages/smartthings-translation-agent/scripts/restore_*.py
agent-packages/smartthings-translation-agent/tests/test_es_co_*.py
```

결과:

- **`es_co_*.py` 52개가 untracked다.** 사고 스크립트를 포함해 커밋 이력·리뷰·blame이 없다.
- **`tests/test_es_co_*.py` 4개도 untracked다.** 전체 테스트 27개 중 23개만 추적된다.
  여기에는 파일럿 A·B의 테스트가 **둘 다** 포함된다
  (`test_es_co_global_merge_260825.py`, `test_es_co_regional_build_260825.py`).
- 따라서 §7이 확인한 로컬 베이스라인 **184 passed는 fresh clone에서 재현되지 않는다.**

이 규칙들은 사고 대응 커밋 `87f0d40` 자신이 `.gitignore`에 15줄을 추가하며 만들어졌다.
공용 코드는 강화하면서 위험한 일회용 코드는 버전 관리 밖으로 밀어낸 셈이다.

### 계획에 미치는 영향

| 계획 항목 | 영향 |
|---|---|
| §7 보완 4 — 신규 writer lint | CI는 untracked 파일을 보지 못한다. lint를 CI에만 두면 정확히 위험한 52개를 놓친다. **로컬 pre-commit 또는 작업 디렉터리 스캔 방식이어야 한다.** |
| §7 보완 4 — 33개 allowlist 동결 | 33개 중 다수가 untracked다. allowlist 파일 자체는 추적하되, 대상이 저장소에 없다는 점을 전제해야 한다. |
| §7 보완 3 — 회귀 fixture | 사고 fixture를 `test_es_co_*.py`로 만들면 **자동으로 ignore된다.** 회귀 테스트는 추적되는 이름으로 만든다. 더 중요하게는 **이미 작성된 회귀 테스트 4건이 저장소 밖에 있다**(아래). |
| 파일럿 A·B | 두 테스트 모두 untracked. 대상 스크립트가 untracked이므로 이는 일관된 결과다. 해법은 아래 참조. |

### 무시되는 테스트가 실제로 담고 있는 것

무시된 4개 중 2개의 테스트 이름은 다음과 같다.

```
test_es_co_global_merge_260825.py
  test_rich_text_replacement_keeps_run_fonts
  test_rich_text_replacement_crosses_run_boundary
  test_cross_workbook_sheet_copy_drops_invalid_column_style_index

test_es_co_regional_build_260825.py
  test_story_051_keeps_existing_row_layout
  test_deleted_regional_sections_remove_physical_rows_and_copy_layout
  test_india_006_reorders_shared_rows_and_uses_local_adds_for_co
  test_story_047_flags_es_derived_co_disclaimer
```

**§7 보완 3이 "만들어야 한다"고 한 사고 회귀 테스트 중 최소 4건이 이미 작성돼 있고,
저장소 밖에 있다.** rich-text run 보존, run 경계 교차, KR 051 no-op 무결성, 섹션 물리 행
삭제는 모두 보고된 사고 항목과 직접 대응한다.

### 왜 같이 무시됐는가 — 규칙이 아니라 결합이 문제다

두 테스트는 일회용 스크립트를 직접 import한다.

```python
from es_co_global_merge_260825 import copy_sheet, replace_product_names
from es_co_regional_build_260825 import build_in_memory
```

대상이 untracked인데 테스트만 추적할 수는 없다. 따라서 **현재 ignore 규칙 자체는
일관적이며, 일회용 스크립트를 제외한 판단도 타당하다.** 52개를 추적하면 저장소가 실제로
어지러워지고, 그 스크립트들은 실제로 일회용이다.

문제는 규칙이 아니라 **회귀 지식이 일회용 도구에 결합돼 있다는 점**이다.
`copy_sheet`·`replace_product_names`는 §3에서 확인한 guard 재구현물이다.

### 권고 — `.gitignore`를 고치지 말고 결합을 끊는다

```
재사용 불변식을 책임에 맞는 추적 모듈로 이전
  → mutation primitive는 guard로 이동
  → contract/no-op 규칙은 contract·verifier 모듈로 이동
  → 테스트도 같은 책임의 추적 파일로 이동
  → .gitignore 수정 불필요
  → 일회용 스크립트가 삭제돼도 회귀 테스트는 남는다
```

**§7·§8이 권고한 "merge 스크립트를 guard로 이전"과 회귀 테스트 보존은 같은 작업이다.**
별도 항목이 아니다.

전부 옮길 필요는 없다. 도구보다 오래 사는 규칙만 옮긴다.

| 테스트 | 성격 | 처리 |
|---|---|---|
| rich-text run 보존 / 경계 교차 | primitive 동작 | guard로 이전 → 추적 |
| cross-workbook style index | primitive 동작 | guard로 이전 → 추적 |
| story 051 no-op 무결성 | contract/verifier 정책 | contract·verifier로 이전 → 추적 |
| 섹션 물리 행 삭제 + layout 복사 | primitive 동작 | guard로 이전 → 추적 |
| india 006 행 재배치 | 도구 고유 로직 | ignore 유지 |
| story 047 disclaimer 플래그 | 도구 고유 로직 | ignore 유지 |

새로 만드는 사고 회귀 테스트도 같은 원칙을 따른다. **일회용 스크립트 이름을 물려받지 말고
실패 유형 이름을 쓴다** (`test_workbook_mutation_guard.py` 또는 추적되는
`tests/test_workbook_regression_*.py`). 테스트는 그것을 낳은 도구보다 오래 살아야 한다.

### 사고 아카이브의 실체 — 도구도 사고 파일도 아니다

`tests/` 전체에 `.xlsx` 바이너리 fixture가 **0개**다. 무시된 테스트들은 입력을 전부 코드로
합성한다.

```python
make_book(base_path)                                   # 최소 워크북 합성
base["US(미국)"].row_dimensions[12].height = 22         # 사고 조건을 코드로 심는다
regional_source["US(미국)"].row_dimensions[12].height = 77
...
self.assertEqual(book["US(미국)"].row_dimensions[12].height, 22)   # 불변 주장
```

따라서 아카이브해야 할 것은 다음 둘뿐이다.

1. **최소 합성 입력 빌더** — 실패 조건을 재현하는 최소 워크북을 만드는 코드
2. **불변 주장** — 그 조건에서 무엇이 유지/변경돼야 하는가

일회용 도구도, 사고 당시의 실제 워크북도 보존 대상이 아니다. **이것이 학습 루프를 성립시키는
저장 형태이며, 이 저장소에는 이미 확립돼 있다.**

**§7 보완 3의 리스크는 이로써 해소된다.** "사고 당시 입력 워크북을 확보해야 하고, 없으면
합성해야 하는데 별도 작업량"이라고 평가했으나, 합성이 이미 표준 방식이므로 실제 사고 파일
확보는 불필요하다.

### 부수 효과 — 고아 테스트가 곧 구현 사양이다

`test_story_051_keeps_existing_row_layout`이 검증하는 불변("no-op 파일은 base의 행 레이아웃을
유지한다")은 **guard에 개념 자체가 없다.** 도구 수준에서 임시로 검증되고 있을 뿐이며,
이것이 §7이 제안한 contract의 `no_op_files`에 해당한다.

즉 저장소 밖에 있는 4개 테스트를 읽는 것이 **guard와 contract에 무엇을 넣어야 하는지에 대한
사양서** 역할을 한다. 사고 아카이브는 학습 루프의 출력이자 다음 구현의 입력이다.
구현 착수 시 가장 먼저 읽을 자료다.

## 10. 최종 설계 경계 — guard·contract·verifier·run

사고 회귀 규칙을 모두 `workbook_mutation_guard.py`에 넣으면 공용 모듈이 다시 결합 지점이 된다.
구현은 파일 수가 아니라 책임을 다음 네 층으로 분리한다. 초기에는 작은 모듈로 시작하고, 실제
코드량이 적으면 contract와 verifier를 한 파일에 둘 수 있으나 책임 경계는 유지한다.

| 책임 | 권장 모듈 | 포함하는 것 | 포함하지 않는 것 |
|---|---|---|---|
| mutation primitive | `workbook_mutation_guard.py` | rich-text 복사, 행·열 안전 삭제, 임시 저장, atomic replace | story 번호·no-op 정책, 작업 상태 |
| 실행 계약 | `workbook_contract.py` | 3축 정본, 위치·속성 단위 `allowed_diffs`, `no_op_files`, 작업별 경로 | 실제 워크북 변경 |
| 독립 검증 | `workbook_verifier.py` | 5축 snapshot, 상세 diff, contract 대조, 검증 레코드 | 생성기의 in-memory expected 재사용 |
| 실행 상태 | `workbook_run.py` | staging, state, events, promote, resume | Excel 세부 복사 로직 |

### 검증 레코드는 artifact와 contract에 결합한다

`promote_verified()`가 단순히 `{"status": "passed"}`를 받으면 오래된 검증 결과나 다른 파일의
레코드를 재사용할 수 있다. 검증 레코드는 최소한 다음을 포함한다.

```json
{
  "status": "passed",
  "staged_file_sha256": "...",
  "contract_sha256": "...",
  "input_authority_sha256": {
    "values": "...",
    "structure": "...",
    "formatting": "..."
  },
  "verifier_version": "...",
  "verified_at": "...",
  "checks": {
    "values": "passed",
    "layout": "passed",
    "styles": "passed",
    "rich_text": "passed",
    "annotations": "passed"
  }
}
```

승격 직전에 `promote_verified()`는 staged 파일과 contract의 SHA-256을 다시 계산한다. 현재
해시가 검증 레코드와 다르거나 필수 check가 하나라도 빠지면 승격을 거부한다. 이는 검증 후
파일이 바뀌거나 입력 정본이 교체된 상태에서 오래된 통과 기록을 사용하는 것을 막는다.

### 검증 레코드 — 구현 전 확정할 5가지

위 스키마는 방향이 옳으나 다음이 미정이며, 그대로 두면 구현자마다 달라진다.

**(1) `checks`의 `passed` 의미를 명시한다.** "diff 없음"이 아니라 **"관측된 diff가 contract의
`allowed_diffs` 선언과 정확히 일치"**다. 전자로 읽으면 허용된 변경이 있는 작업은 영원히
통과하지 못한다. 각 축은 `passed` / `unexpected_diff` / `missing_declared_diff` 중 하나를
갖고, 뒤의 둘은 §7 보완 1의 `unexpected` / `missing`과 같은 구조로 상세를 담는다.

**(2) 입력 정본 해시도 승격 직전에 재계산한다.** 현재 재계산 대상은 staged 파일과 contract
둘뿐이다. `input_authority_sha256`을 기록만 하고 검사하지 않으면 그 필드는 장식이며,
원 제안의 실패 유형 `input_drift`(작업 도중 원본이 바뀜)를 잡지 못한다. 세 정본 모두
승격 직전에 재확인하고 불일치 시 거부한다.

**(3) 정본이 디렉터리인 경우를 정의한다.** 값 정본은 보통 단일 파일이 아니라 33개 워크북이
든 폴더다. `input_authority_sha256`은 스칼라가 아니라 **정렬된 `(상대경로, sha256)` 목록의
해시**로 정의하고, 목록 자체도 레코드에 남긴다. 파일이 추가·삭제된 경우도 drift로 잡힌다.

**(4) 경로는 설정 가능하되 분리는 불변식이다.** contract가 `staging_root`·`delivery_root`를
선언하는 것은 옳지만, 같은 값으로 선언하면 승격 관문이 사라진다. run 계층은 두 root가
서로 다르고 어느 쪽도 상대의 하위 경로가 아님을 검사한 뒤에만 진행한다. delivery_root로의
쓰기는 `promote_verified()` 외에 존재하지 않는다.

**(5) `verifier_version` 불일치는 거부한다.** 경고가 아니라 거부다. verifier가 바뀌었다면
이전 레코드의 통과 판정은 현재 기준의 통과를 의미하지 않는다. 재검증 후 승격한다.

### 경로 호환성

staging/delivery root는 전역 상수로 고정하지 않는다. 기존 작업은 사용자가 지정한 다양한 외부
폴더를 사용하므로 contract가 이번 작업의 `work_root`, `staging_root`, `delivery_root`를 선언한다.
guard와 run 계층은 이 선언을 해석하고, 기존 산출물 형제 manifest의 위치·이름은 바꾸지 않는다.

### ignored writer의 이전 원칙

52개 일회용 writer 전체를 추적 대상으로 바꾸지 않는다. 대신 재사용되는 안전 로직과 회귀
불변식은 위 추적 모듈·테스트로 승격하고, 업무 고유 옵션만 얇은 ignored wrapper에 남긴다.
리팩터링 전에 ignored 사고 스크립트와 테스트의 파일 목록·SHA-256을 기록해 Git으로 복구할 수
없는 현재 상태를 보존한다.

## 요약

제안은 "지침서를 시스템으로 바꾸자"고 말한다. 그러나 이 패키지는 이미 시스템의 부품을
갖추고 있고, **부품을 쓰지 않고도 납품이 되는 것**이 문제다. 그래프를 얹기 전에 우회로를
먼저 막아야 효과가 난다. 제안 중 실행 시점 계약(`contract.json`)과 실행 원장은 진짜 공백이며
채택할 가치가 있다.

후속 마이그레이션 계획(§7)의 전략 — 추가 후 병렬 비교, 신규부터 강제, 위험도 순 이전 —
은 타당하다. 다만 구현 착수 전에 세 가지를 정정해야 한다. **(1) 저장 함수를 두 개로
늘리지 말 것**(호출부가 3곳뿐이며, 문을 두 개 만드는 것이 이 사고의 원인 구조다),
**(2) 기존 manifest를 per-work 디렉터리로 옮기지 말 것**(현재 규약은 형제 파일이며 옮기면
소비 코드가 깨진다), **(3) 사고 스크립트를 동등성 검증 대상으로 삼지 말 것**(기존 출력이
이미 틀렸으므로 무차이를 조건으로 걸면 버그가 사양이 된다).

CLI 안(§8)도 사용성 개선으로는 타당하다. 그러나 승격 독점을 CLI로 구현할 수는 없다.
"도구를 하나의 입구로 묶는" 통합은 `/st-apply`에서 이미 수행됐으나 **산문 계층에만
있었기 때문에** 사고를 막지 못했다. 따라서 답은 네 번째 산문 입구가 아니라 기존 입구들이
공통으로 호출하는 **결정론적 실행기**다.

귀속에 관해서는 초판을 정정했다(§8). git 이력 확인 결과 guard와 그 지침은 사고
**다음 날** 만들어졌다. 사고는 지침 불이행이 아니라 **당시 공용 안전 기반과 실행 계약이
없었던 것**이 원인이며, 이 문서가 제안하는 작업은 재발 방지가 아니라 2026-08-26에 시작된
대응의 미완 부분 완결이다. 이 구분이 결과물을 바꾼다 — 불이행 문제로 보면 lint만 만들게
되고, 기능 부재 문제로 보면 `delete_cols_with_manifest`·contract·상세 verifier를 만들게 된다.

강제 수단에 관해서도 초판의 과장을 정정했다. guard 역시 호출하지 않으면 무력하다.
정확한 표현은 "절대 강제"가 아니라 **"저장소 수준의 구조적 강제"**이며,
guard·lint·allowlist·회귀 테스트를 함께 써야 성립한다. 그리고 §9가 보여주듯 이 저장소에서는
"저장소 수준"조차 전제가 흔들린다 — 정작 위험한 52개가 git 밖에 있다.

최종 구현 경계는 §10과 같다. mutation primitive, 실행 계약, 독립 verifier, 실행 상태를
분리하고, 검증 레코드를 staged artifact·contract·입력 정본의 해시에 결합한다. CLI가 있더라도
새 사용자 입구가 아니라 기존 `/st-edit`·`/st-apply`가 호출하는 공통 실행기로 둔다.

## 구현 에이전트를 위한 착수 체크리스트

0. **§9를 먼저 읽고 ignored 파일을 보존한다.** `es_co_*.py` 52개와
   `test_es_co_*.py` 4개의 목록·SHA-256을 기록한다. lint 배치 위치와 회귀 테스트 파일명이
   여기에 달려 있다.
1. `python3 -m pytest tests/ -q` 실행 — 로컬 184 passed / 16 subtests 확인.
   단 이 중 4개 테스트는 untracked이므로 fresh clone 기준이 아니다.
2. **저장소 밖의 회귀 테스트 4건을 먼저 읽는다** (§9). 이것이 guard·contract 구현 사양이다.
   실제 사고 워크북은 필요 없다 — 아카이브는 최소 합성 빌더 + 불변 주장이다.
   회귀 테스트는 **추적되는 파일명**으로 만든다.
3. §10의 책임 경계를 확정한다. 최소 구현은 guard / contract·verifier / run으로 합칠 수 있지만
   no-op 정책과 작업 상태를 mutation guard에 넣지 않는다.
4. contract 스키마 정의 — `allowed_diffs`를 축이 아닌 **위치·속성 단위**로 선언하고,
   작업별 `work_root` / `staging_root` / `delivery_root`를 포함한다. 값 축은
   `verify_global_output`(:284-291)의 셀 단위 대조를 참고 구현으로 삼는다.
5. `delete_cols_with_manifest` 추가 (§3 결함 3) — 대응 당시 빠진 기능.
6. `save_verified_atomic`에 contract를 연결하고, 상세 verifier가 artifact·contract·입력 정본
   해시를 포함한 verification record를 생성하도록 한다. 호출부 3곳을 수정한다.
7. `promote_verified()`는 승격 직전에 staged artifact·contract·**입력 정본** 해시를 모두
   재계산하고(§10 (2)(3)), `verifier_version`이 일치하며 필수 check 전부가 contract 선언과
   일치할 때만 contract의 delivery root로 승격한다. staging/delivery root가 동일하거나
   포함 관계면 그 전에 거부한다(§10 (4)).
8. 재사용 불변식을 추적되는 회귀 테스트로 이전한 뒤 33개 allowlist 동결 + 신규 writer lint를
   적용한다 (§7 보완 4, §9 제약 반영).
9. 파일럿 A(`es_co_regional_build_260825.py`, **무차이가 성공**) → 회귀 확인.
10. 파일럿 B(`es_co_global_merge_260825.py`, **차이가 성공**) → 결함 수정 확인 (§7 정정 3).
   기준선은 기존 출력이 아니라 입력 정본 + contract에서 독립 계산한 기대 결과다.
11. `/st-edit`·`/st-apply`가 공통 실행기를 호출하도록 연결 (§8).
12. 실제로 재실행되는 기존 writer부터 순차 이전.

## 근거 파일

- `agent-packages/smartthings-translation-agent/scripts/workbook_mutation_guard.py`
- `agent-packages/smartthings-translation-agent/scripts/es_co_global_merge_260825.py`
- `agent-packages/smartthings-translation-agent/scripts/es_co_regional_build_260825.py`
- `agent-packages/smartthings-translation-agent/scripts/batch_co_rollout.py`
- `agent-packages/smartthings-translation-agent/references/excel-workflow.md`
- `agent-packages/smartthings-translation-agent/tests/test_workbook_mutation_guard.py`
- `agent-packages/smartthings-translation-agent/tests/test_batch_resume.py`
- `agent-packages/smartthings-translation-agent/commands/README.md`
- `agent-packages/smartthings-translation-agent/commands/st-apply.md`
- `agent-packages/smartthings-translation-agent/commands/st-edit.md`
- `.gitignore` (55-66행)
- git 커밋 `87f0d40` "Harden workbook mutation and manifest validation" (2026-08-26 10:53)
- `docs/agent_architecture_notebooklm_assessment_20260811.md`
