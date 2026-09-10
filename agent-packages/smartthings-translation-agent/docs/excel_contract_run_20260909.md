> 후속 리뷰 수정: 증분 하이라이트 계약 경유, 발행 이력과 현재 검증기 구분,
> 신규 writer AST 검사와 해시 측정을 반영했다. 최신 결과는 **269 passed, 22 subtests**.
> 상세: `docs/excel_post_step5_review_20260909.md`. 아래 260개 결과는 5단계 완료 당시 기록이다.

# 5단계 완료: 기존 명령의 기본 경로 연결 (2026-09-09)

사용자 명령은 `st-start`, `st-ask`, `st-translate`, `st-inspect`, `st-edit`, `st-apply`의
6개로 정리했다. `st-review`/`st-help`/`st-pipeline`은 호환 경로다. 상세 정본은
`commands/README.md`와 `references/command-execution.md`다.

`workbook_apply_edits.py` 기본 경로를 실행기로 전환하고, `workbook_delivery.py`에서 기존
story/review apply를 같은 실행기에 연결했다. 새 실행기·DB·역할은 추가하지 않았다.
기존 검수 승인 변환, 편집 preflight, revision/outcome 형식, 앱 glossary/occurrence resolver,
기존 rich-text renderer를 재사용한다. 납품용 resolver는 모델/RAG client를 생성하지 않는다.

- 같은 요청은 동일 작업 계약과 journal로 재개하고 완료 결과를 재사용한다.
- draft와 final을 하나의 검증 배치로 공개한다. 원본과 기존 버전은 유지한다.
- `--dry-run`은 Excel을 쓰지 않고 계획만 저장한다. 실행은 기존 사용자 승인 경계다.
- source·glossary·승인 manifest·활성화 manifest·로컬 의존 코드/규칙을 해시로 고정한다.
- 값 기대치는 승인 편집에서, 열 삭제 기대치는 원본 좌표의 독립 투영에서 계산한다.
  glossary 계획은 해당 시트·범위의 rich text만 변경할 수 있다. 실제 저장본은 독립 verifier가
  정본과 정확한 허용 diff를 다시 비교한다. 검증 실패 시 허용 범위를 확장하지 않는다.
- native review는 accept/partial/hold를 유지하고 report는 approved만 반영한다. 판정 완료 행은
  work_id를 붙여 기존 outcome JSONL 형식으로 기록하며 pending은 승인/거절로 추정하지 않는다.
- `--output`의 실제 산출물은 그 부모의 `verified/<work_id>/`에 생성된다. 반환된 final 경로가
  납품 경로다. 이름만 지정한 예전 경로를 사용하지 않는다.

메모 상자의 삭제·이동(VML)은 별도 서식 계획을 요구하고 자동 처리하지 않는다.
복잡한 구조 planner·과거 일회용 writer 전면 이전·전역 save lint는 이번 범위 밖이다.
단독 고급 highlight와 API 번역 writer에는 이 배치 보장을 소급 적용하지 않는다.
아래 2~4단계 기록은 구현 경과이며 opt-in 설명은 당시 상태다.

## 5단계 검증 결과

`../../venv/bin/python -m pytest tests -q -p no:cacheprovider`:
**260 passed, 22 subtests passed**, 외부 라이브러리 deprecation warning 2건.
신규 `test_workbook_delivery.py`의 10개 통합 사례와 기존 250개 테스트를 포함한다.
원본 불변·기본 CLI·재개·drift 차단·미승인 제외·배치 실패·열 이동/병합/서식 보존을 확인했다.
기존 inventory의 56개 파일 SHA-256도 모두 일치한다. 합성 Excel만 사용했고 유료 API 호출과
사용자의 실제 납품 파일 생성은 하지 않았다. 실사용 시 구조/서식 변경 대표본의 시각 검토는
기존 Excel 0단계 지침대로 수행한다.

---

# Excel 계약 실행 파일럿 — 2~4단계

작성일: 2026-09-09
범위: 실행 계약, 독립 검증, 일반 셀 편집 writer 파일럿, 상태·이벤트·재개·제한 복구.
상위 문서: app repo `docs/excel_mutation_loop_graph_assessment_20260827.md`.

## 설계 범위와 과도한 하네스 방지

새 프레임워크·DSL·DB·데몬·대시보드·에이전트 역할을 도입하지 않았다. 표준 라이브러리와
기존 openpyxl로 구현했다. 계약은 JSON 데이터, 실행은 `execute_run()` 함수 하나다.
세 신규 모듈은 책임을 나누기 위한 것으로 plugin/adapter registry나 범용 scheduler가 아니다.
실행 원장은 작업 단위의 작은 파일이며 스트리밍 이벤트 플랫폼을 목표로 하지 않는다.

문서의 과거 사고 writer A/B 전체 이전 대신, 최신 5단계 계획의 3단계인 **대표 writer 하나**를
추적되는 `workbook_apply_edits.py`에서 검증했다. 기본 호출은 유지하고 명시적인 계약 경로를
추가했다. `/st-edit`·`/st-apply` 전체 진입점 전환, legacy writer allowlist/lint 강제,
감수·번역 전체 파이프라인 연결은 5단계 후속 범위다. 현재 guard를 호출하지 않는 코드를
OS 수준에서 차단한다고 주장하지 않는다.

## 구현 책임

| 코드 | 책임 |
|---|---|
| `scripts/workbook_contract.py` | 3축 정본·파일/디렉터리 해시 목록·정확한 허용 diff·no-op·경로·복구 한도 검증 |
| `scripts/workbook_verifier.py` | 입력 정본을 디스크에서 독립 재개방, 5축 hash 비교 후 상세 diff, 검증 레코드 |
| `scripts/workbook_run.py` | 실행 승인 확인, 상태·이벤트 기록, 재개, 제한 복구, 배치 전체 검증 후 공개 |
| `scripts/workbook_mutation_guard.py` | 기존 저장 함수의 contract 옵션, 공용 상세 snapshot, 안전한 열 삭제와 audit |
| `scripts/workbook_apply_edits.py` | 기존 preflight를 재사용하는 계약 preview·실행/재개 파일럿 |

`save_verified_atomic`는 하나만 유지한다. contract를 전달한 저장은 staging 안으로 제한하고
5축 통과·파일 해시·계약 내용 해시가 맞는 레코드를 요구한다. 기존 contract 없는 호출의 호환은
유지되며, 그 호출이 새 실행기의 전체 보장을 받는 것은 아니다.

## 계약 데이터

`schema_version=1`, `work_id`, `work_root`, `staging_root`, `delivery_root`,
`authorities`, `writer`, `artifacts`, `no_op_files`, `recovery`를 저장한다.
`writer`는 ID·코드 버전·`cost=local`을 갖는다. 이 runner는 API 비용이 드는 writer를 받지 않는다.

각 artifact는 세 정본을 별도로 참조한다. 정본은 authority ID와 해당 inventory의 상대 파일명으로
지정한다. 단일 파일은 `file="."`이다. 디렉터리는 정렬된 상대경로·SHA-256 목록과 목록 자체의
해시를 저장하며, 파일 추가·삭제·내용 변경도 입력 변경으로 판정한다. 용어집 등 워크북 이외의
의존 파일도 추가 authority로 묶을 수 있다. 모든 authority는 실행/납품 root와 분리한다.

축의 정본 매핑은 명시적이다: values → 값 정본, layout → 구조 정본,
styles/rich_text/annotations → 서식 정본. 행 높이는 layout 축에 속한다. 별도 높이 변경은
정확한 허용 diff로 선언한다. `no_op_files`는 artifact ID 목록이며 하나의 동일한 정본과
0개의 허용 diff를 요구한다.

허용 diff 예:

```json
{
  "axis": "layout",
  "path": ["sheets", "US", "row_dimensions", "5", "height"],
  "before": {"present": true, "value": 17},
  "after": {"present": true, "value": 21}
}
```

축 전체 wildcard는 없다. before는 verifier가 독립적으로 연 정본과 일치해야 한다.
관측된 변경 집합과 선언된 변경 집합이 정확히 같아야 통과한다. 빠진 변경은 `missing`,
추가·다른 변경은 `unexpected`로 남긴다. 검증기는 생성기의 in-memory expected를 받지 않는다.

검증 레코드는 artifact ID·산출물 SHA-256·계약 파일/내용 해시·모든 입력 inventory·verifier
코드/openpyxl 버전·5축 checks·검증 시각을 담는다. 승격 직전 모두 재확인하며 변경된 레코드를
재사용하지 않는다. 상태의 단순 completed나 파일 존재 여부는 검증 근거가 아니다.

## 실행과 재개

```text
contracted → awaiting_approval → ready → generating → verifying → verified
                                           ↑             ↓
                                           └─ recovering
verified (전체 파일) → promoting → completed
오류·입력 변경·반복 한도 → blocked / failed
```

실행 파일은 기존 산출물 형제 manifest와 별개다:

```text
<work_root>/contract.json
<work_root>/events.jsonl          # 원장: 전이·시도·실패·다음 행동
<work_root>/state.json            # 최신 상태 cache
<work_root>/verification/*.json
<work_root>/failures/*.json
<work_root>/edit_result.json      # 파일럿 revision 중복 생성 방지
<staging_root>/<artifact.xlsx>
<delivery_root>/<work_id>/<artifact.xlsx>
<delivery_root>/<work_id>/batch.complete.json
```

새 실행의 staging은 비어 있어야 한다. journal을 atomic 교체한 뒤 state cache를 쓴다.
cache 저장 전에 프로세스가 죽어도 다음 호출이 journal에서 상태를 읽는다. OS 파일 잠금은
동시 실행을 막으며 프로세스 종료 시 풀린다. PID 파일이나 잠금 파일의 존재로 실행 중이라고
추정하지 않는다. 이 구현은 macOS/Linux의 `fcntl`을 사용하며 Windows 지원은 포함하지 않는다.

재개 시 입력·계약·writer를 다시 검사한다. staging이 있으면 독립 재검증하고 생성은 건너뛴다.
verifier 변경도 재검증 결과에 반영된다. 이미 공개된 배치가 있으면 해시·버전·출력 경로를 확인해
완료 상태를 복원한다. 현재 기준과 맞지 않는 기존 공개본은 덮어쓰지 않고 중단한다.
실패 한도와 동일 실패 횟수는 journal에 남아 재호출해도 초기화되지 않는다.

## 제한 복구 정책

| 실패 | 행동 |
|---|---|
| layout/styles/rich_text/annotations 불일치 | 고정된 원본에서 다시 생성하고 독립 검증 |
| EAGAIN/EBUSY/ETIMEDOUT 로컬 I/O | 같은 생성 단계 제한 재시도 |
| 값 변경 불일치, 누락된 값 변경 | 자동 수정/재승인 없이 중단 |
| 입력·용어집 등 dependency 해시 변경 | 중단 후 새 정본 snapshot·계약·승인 필요 |
| contract 또는 writer 버전 변경 | 새 run과 승인 필요 |
| 동일 실패 2회 또는 총 생성 3회(기본) | failed; 재호출로 예산을 초기화하지 않음 |
| 분류되지 않은 오류·미지원 콘텐츠 | 중단 후 검토 |

복구 시 allowed_diffs나 입력 해시를 자동 갱신하지 않는다. 동일한 deterministic writer에서
같은 결함이 반복되면 한 차례 재생성 후 중단한다. 모델 호출이나 프롬프트 재시도는 없다.

## 배치 공개와 기존 이력

한 파일씩 최종 경로를 덮어쓰지 않는다. delivery root의 숨겨진 임시 디렉터리에 검증된 파일을
복사하고, 복사본·입력·계약을 재확인한 뒤 배치 디렉터리를 한 번에 rename한다. staging과
delivery가 다른 볼륨이어도 복사 후 destination-local rename을 사용한다. 소비자는 반환된
버전 디렉터리와 `batch.complete.json`만 사용해야 한다. 중간 파일 탐색은 완료 판정이 아니다.

일부 파일 검증 실패나 공개 중 프로세스 중단은 이전 버전을 바꾸지 않는다. 강제 종료로 남은
숨겨진 `.preparing-*` 디렉터리는 공개본이 아니며 자동 GC는 이번 구현에 추가하지 않았다.
`fsync`와 atomic rename을 사용하지만 모든 네트워크 파일시스템/하드웨어 장애를 검증한 것은 아니다.

파일럿의 revision과 `.changes.json`은 기존 형식을 사용한다. 반복 완료 호출은 같은 revision을
돌려준다. 워크북 공개 뒤 legacy 이력 기록이 완료되기 전에 중단되면 다음 호출이 이력을 만든다.
이력 생성과 cache 기록 사이의 강제 종료에서는 고아 revision이 남을 수 있으나 워크북은 중복
적용되지 않는다. 별도 이력 트랜잭션 시스템은 만들지 않았다.

**일반 편집 산출물은 여전히 draft다.** `delivery_root`는 실행기의 검증된 출력 경계 명칭이며,
업무상 glossary 전체 재하이라이트가 끝난 최종 번역 납품본이라는 뜻이 아니다. 실제 납품은
기존 `/st-apply` 절차를 따른다.

## 파일럿 사용법

패키지 실행 루트에서, workbook은 변경되지 않는다. 명시적인 새 work_id를 사용한다.

```bash
# 1. 계약 preview: Excel 쓰기 없음 (contract.json만 저장)
python scripts/workbook_apply_edits.py /path/source.xlsx /path/edits.json \
  --prepare-run /path/work/pilot-001 --delivery-root /path/verified \
  --work-id pilot-001 --json

# 2. 표시한 변경과 계약 해시를 사용자가 승인한 후에만 approval.json 준비
# {"approved":true,"approved_by":"사용자","contract_sha256":"preview에 표시된 해시"}

# 3. 실행; 중단 후에는 같은 명령으로 재개
python scripts/workbook_apply_edits.py /path/source.xlsx /path/edits.json \
  --run-contract /path/work/pilot-001/contract.json --approval /path/approval.json --json

# 상태만 보기 (생성·복구 실행 없음)
python scripts/workbook_apply_edits.py /path/source.xlsx /path/edits.json \
  --run-contract /path/work/pilot-001/contract.json --dry-run --json
```

숨김/보호/수식/병합 예외 플래그는 prepare 시 승인할 계획에 포함한다. 실행 시 변경한 플래그로
기존 계약을 바꾸지 않는다. 임의 사용자 인증·서명 시스템은 만들지 않았으며 승인 파일은
명시적인 사람 승인 사실을 기록하는 로컬 manifest다.

## 지원 범위

5축 상세 정보에는 셀 값·수식, 행/열 geometry·스타일, 시트 순서·병합·보호·인쇄 속성,
rich-text run·폰트, 주석·하이퍼링크, 유효성 조건·범위, 조건부 서식·table·이름 정의를 포함한다.
그림/차트/anchor는 패키지 XML과 바이너리 hash로 비교하므로 개수만 같은 이동도 탐지한다.

VBA·ActiveX·pivot·slicer·외부 링크·embedded object·connection과 openpyxl이 제거한다고 경고하는
확장 객체는 자동 통과시키지 않고 거부한다. OOXML 전체 편집 엔진을 새로 구현하지 않는다.
열 삭제 primitive는 병합 경계·dimension group 교차와 재배치할 수 없는 수식/객체 의존성을
변경 전에 거부하고, 지원되는 열 삭제에서는 메모·서식·너비·병합 이동을 기록한다.

파일럿은 기존 사용 영역 안의 JSON scalar 셀 편집이다. 새 행/열·시트 구조 변경 planner,
전체 시트 복사, 자동 glossary renderer는 포함하지 않는다. 이런 작업의 생성기는 후속에
연결할 수 있으나 정확한 계약과 별도 검증을 충족해야 한다.

## 검증 및 완료 근거

`tests/test_workbook_contract_run.py`는 합성 입력에서 다음을 검사한다.

- 3축 정본 분리, no-op, 위치·속성별 허용/미허용/누락 변경, 5축 검증
- 입력 디렉터리 추가·삭제·변경, 계약·승인·writer·산출물·레코드 불일치 거부
- 기존 writer 파일럿 실행·반복 재개·원본 불변·수식·비대상 rich text·revision 유지
- 형식 오류 복구, 동일 오류 중단, 총 시도 제한, 값 오류 즉시 중단, local I/O 복구
- staging 저장 후 중단, 실제 별도 프로세스 강제 종료 후 재개, 동시 프로세스 잠금
- 배치 일부 실패·복사 중 중단·공개 직후 중단 복원, 중복 적용 방지, 공개 metadata 경로 검증
- chart anchor 이동, data validation 조건 변경, 열 삭제의 의존성·병합·dimension 안전성

테스트 워크북만 임시 폴더에 생성했으며 사용자 업무 워크북이나 API는 실행하지 않았다.
최종 실행 결과:

- 새 계약·실행 테스트: **52 passed**.
- 전체 패키지: **250 passed, 22 subtests passed**, 6.69초.
- 기존 외부 의존성의 deprecation 경고 2건만 남았다.
- 명령: `../../venv/bin/python -m pytest tests/ -q -p no:cacheprovider`.
- 독립 폴더 검증: app repo·ignored writer 없이 필요한 Python 파일 6개와 신규 테스트만
  임시 폴더에 복사하여 **52 passed**. 기존 venv의 라이브러리를 사용한 코드 의존성 격리이며,
  라이브러리 신규 설치나 Windows 검증은 아니다.
- `git diff --check` 통과. 테스트는 업무 입력과 분리된 합성 파일만 사용했다.

완료 감사:

| 요청 단계 | 구현 증거 | 검증 증거 |
|---|---|---|
| 2. contract·독립 verifier | `validate_contract`, `verify_artifact`, `validate_record` | 3축·5축·정확한 diff·no-op·해시/버전·directory drift 테스트 |
| 3. 기존 writer 하나 연결 | `prepare_edit_contract`, `apply_edits_contract`, 기존 CLI 옵션 | 원본 불변·정상 셀/수식/rich text·별도 CLI 실행·반복 revision 동일 테스트 |
| 4. 상태·이벤트·재개 | `execute_run`, `events.jsonl`, `read_state` | 프로세스 강제 종료·cache 누락·staging 재검증·공개 직후 재개·동시 실행 차단 |
| 4. 실패별 제한 복구 | `recovery_decision`, 지속되는 attempts/failures | 형식 실패 복구·같은 실패 2회·전체 3회·값 오류 중단·로컬 I/O 복구 |
| 납품 경계 | `promote_verified`, 완료 manifest와 단일 batch rename | 일부 실패/복사 중 중단 시 미공개·입력 drift·출력/manifest tampering 거부 |

현재 작업은 위 2~4단계를 완료한 파일럿이다. 전체 writer 이전·기본 진입점 전환을 완료했다고
확장해서 해석하지 않는다. Git staging/commit은 실행하지 않았다.
