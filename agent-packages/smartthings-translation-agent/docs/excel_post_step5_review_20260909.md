# 5단계 구현 리뷰 반영 (2026-09-09)

리뷰의 1~3번 결함·누락은 확인 후 수정했다. 4번은 합성 배치로 측정했다.
다만 “증분 highlight 하나를 고치면 모든 사용자 경로가 닫힌다”는 범위는 채택하지 않는다.
단독 glossary highlight와 API 번역 등 고급 writer의 전면 이전은 여전히 별도 작업이다.

## 1. 증분 검수 하이라이트

기존 save → replace → 텍스트 검증을 제거했다. `workbook_incremental_highlight.py`는
값·구조·서식 정본을 입력 파일로 선언하고, 범위 내 정확한 rich-text 변경 및 기존 공백
hardening 좌표만 계약에 기록한다. 값/구조/일반 스타일/주석 또는 범위 밖 rich-text 변경은
계획에서 거부한다. 공용 runner와 독립 5축 verifier가 staging을 검증하고 버전 배치로 공개한다.
실제 경로는 `verified/<work_id>/`의 반환된 `output`이다. 결정론적 단일 렌더링은 실패 시
자동 반복하지 않는다. 새 그래프나 실행기를 만들지 않았다.

저장 중 행 높이를 바꾸는 실패 주입에서 미공개를 확인했다. 한국어 파일명, 정상 rich-text,
범위 밖 서식 변경 차단도 검증했다. 단순히 replace 순서만 바꾸는 수정은 채택하지 않았다.

## 2. 이미 발행된 배치의 재조정

`validate_record`는 staging/새 발행용으로 현재 verifier 버전과 실제 입력을 계속 확인한다.
`validate_published_record`는 발행 당시 계약 해시·계약 내용 해시·artifact ID·출력 파일 해시·
완전한 5축 통과 기록·계약에 저장된 입력 inventory를 확인한다. 현재 verifier 신원이나
지금의 원본 파일은 역사적 산출물의 동일성 요건에 포함하지 않는다.

승인·계약·발행 manifest를 확인한 뒤 이미 발행된 배치가 있으면 writer/입력 재실행 게이트
앞에서 복원한다. 잘못된 재조정으로 blocked가 된 기존 작업도 동일한 발행물이 확인되면
회복할 수 있다. 파일/계약/기록 훼손은 계속 차단한다. 최신 규칙으로 재검증했다는 뜻은 아니다.
원본이 삭제되고 verifier/writer 버전이 바뀌어도 기존 결과를 재생성 없이 반환하는 테스트와,
미발행 레코드는 구버전 verifier를 계속 거부하는 테스트가 통과했다.

## 3. 신규 우회 증가 탐지

`lint_workbook_writers.py`와 `workbook_writer_baseline.json`을 추가했다.
`scripts/**/*.py`의 save/save_workbook/ExcelWriter/to_excel 사용을 AST로 세고, 기존 경로·
표현별 개수보다 늘어난 사용을 거부한다. guard import만으로 면제하지 않는다. save 메서드를
변수에 넘기는 참조도 탐지한다. 현재 기준은 31개 파일의 35개 참조이며, 안전한 공용 save
primitive도 포함한 수치다. 이것을 “우회 writer 31개”와 같은 수치로 해석하지 않는다.
ignored 과거 스크립트가 없는 fresh clone은 허용하되 새 사용을 자동 등록하는 옵션은 없다.

패키지 전체 pytest에 검사 테스트를 연결했다. 직접 실행은
`python scripts/lint_workbook_writers.py`다. 테스트 실행 없이 OS 수준에서 모든 쓰기를
차단하는 장치는 아니다. 동적 reflection·외부 도구·다른 쓰기 API까지 완전 탐지한다고
주장하지 않는다. `save_verified_atomic(contract=None)`에는 DeprecationWarning을 추가했다.
과거 writer를 일괄 수정하지 않았으며 inventory의 56개 파일 해시는 모두 유지됐다.

## 4. 해시 비용 측정

로컬 임시 디렉터리에 1 MiB 파일 33개와 합성 source.xlsx 1개를 두고 동일 정본을 참조하는
no-op 산출물 33개를 검증·발행했다. contract 준비 시간은 제외하고 runner의 inventory
호출 수와 시간을 계측했다. 유료 API나 실제 워크북은 사용하지 않았다.

| 항목 | 측정값 |
|---|---:|
| 정본 1회 읽기 크기 | 34,607,835 bytes (약 33 MiB) |
| inventory 호출 수 | 200회 |
| 논리적 누적 해시 입력 | 6,921,567,000 bytes (약 6.45 GiB) |
| inventory 누적 시간 | 3.352초 |
| 전체 배치 실행 시간 | 4.498초 |

리뷰의 165회보다 많다. 현재 runner는 artifact당 verify 2회 + guard 1회 + promotion
검사 3회에, 실행 시작과 공개 직전의 전체 검사 2회를 더해 `6N + 2`회 수행했다.
상위 command adapter 및 contract 준비에서는 추가 검사가 있을 수 있다.
한 번의 warm-cache 로컬 측정이므로 네트워크 드라이브·대형 정본의 보장값은 아니다.
현재는 정확성 게이트를 유지한다. 큰 실배치에서 비용이 문제가 되면 공개 전후의 검사
경계를 유지하면서 배치 단위 중복 해시부터 줄이고, 장기 캐시는 도입하지 않는 것이 적절하다.

## 검증

`../../venv/bin/python -m pytest tests -q -p no:cacheprovider`:
**269 passed, 22 subtests passed**, warning 11건(기존 라이브러리 2건 + 의도한 legacy 경고).
writer lint 통과, 기존 56개 회귀 자산 해시 유지, `git diff --check` 통과.


## 2026-09-10 추가 검토: 앱 생성 경계와 납품 게이트

**확인한 현황이며 아래 제안은 아직 구현하지 않았다.** `workbook_translate.py`는
`_app_pipeline.bootstrap_project()`로 앱 src를 연결하고 `TranslationChecker`의 integrated
pipeline을 호출한다. 앱 `checker_service.py:1371,1613`에서 워크북을 직접 저장하고 그 뒤
텍스트를 검사한다. 패키지 event_summary의 `status: ok`와 `excel_path`는 앱 실행 완료를
뜻하며, 패키지 계약에 따른 최종 납품 검증 완료를 뜻하지 않는다.

| 사용자/도구 경로 | 실제 생성 책임 | 현재 계약 경계 |
|---|---|---|
| st-translate / st-pipeline translate → workbook_translate | 앱 checker_service | 앱 산출물에 패키지 5축 납품 게이트 없음 |
| st-highlight → workbook_highlight_glossary | 앱 highlight_only + 패키지 보존 보정 | 일부 guard 보정만 있음; 전체 계약 납품과 다름 |
| st-textbook → text_workbook_create | 앱 services/text_workbook_service | 앱의 워크북 생성 경로, 패키지 계약 없음 |
| st-pipeline audit → workbook_audit | 앱 inspection | 검수 리포트 경로; translate의 워크북 쓰기와 구분 |
| st-edit / st-apply / 증분 review highlight | 패키지 공용 runner | 선언 계약·독립 검증·버전 배치 발행 |

AST 검사는 패키지 scripts의 정적 save 참조만 대상으로 한다. 앱 src의 저장은 검사 범위
밖이다. guard import 여부나 명령 설치 여부로 이 경계를 추론해서는 안 된다. 현재 경계는
앱 repo 자체의 쓰기를 통제하는 보안 sandbox가 아니다.

**권고 설계:** 앱을 수정하지 않고도 현재 소스 경로에서 출력명을 만드는 동작을 이용해,
격리된 작업 폴더의 입력 복사본을 앱에 전달할 수 있다. 앱 생성 결과는 candidate로 취급하고
기존 delivery 디렉터리와 분리한다. API 실행 승인과 납품 검증은 별개이며 검증 실패로
유료 생성 전체를 자동 재실행하지 않는다.

1. API 호출 전 원본·용어집·대상 좌표·구조/서식 정본과 보존 범위를 고정한다. CO prep을
   사용하는 경우 prep 자체의 정본 대비 보존도 검증해야 한다.
2. 생성 후 고정한 원본과 비교해 범위 밖 값, 구조, 서식 손실 및 예상 대상 셀의 누락을
   검사한다. LLM 성공 이벤트와 완료 셀 수만으로 전수 반영을 판정하지 않는다.
3. 번역 후보를 검토하고 승인 범위의 정확한 before/after를 확정한다. 생성 결과 전체 diff를
   allowed_diffs로 자동 채택하면 손실까지 승인하는 순환 검증이 된다.
4. 승인된 값만 고정 원본에 적용하는 기존 st-apply 경로를 우선 재사용한다. 검증되지 않은
   앱 결과를 그 자체로 값/구조/서식의 정본으로 삼지 않는다. 앱 파일을 직접 발행하려면
   별도 확정 계약에 대한 verify_artifact 통과와 기존 배치 승격이 필요하다.

현재 verify_artifact는 정확한 expected after를 요구하는 검사기다. 생성 전 알 수 없는
번역값의 의미적 타당성까지 자동 판단하지 않는다. 따라서 함수 호출 한 번을 붙이는 것으로
파이프라인 납품 검증이 완성된다고 보지 않는다. 새 범용 하네스보다는 candidate 경계와
기존 승인·납품 경로 연결에 한정하는 것이 적절하다.

## 2026-09-10 추가 검토: 잔여 writer 목록과 이전 순서

현재 AST baseline의 31개 파일에는 `bootstrap.py`의 CLI `args.save` 참조(워크북 저장 아님)와
공용 저장 primitive `workbook_mutation_guard.py`가 포함된다. 이를 제외한 직접 save 참조가
있는 파일 29개는 아래와 같다. 목록은 동적/외부 writer를 포함하는 완전한 보안 목록이 아니다.

- `scripts/backtranslation_restore_kr_richtext.py`
- `scripts/batch_co_rollout.py`
- `scripts/combine_kr_us.py`
- `scripts/create_039_first_acceptance.py`
- `scripts/create_us_fixed_copy.py`
- `scripts/es_co_agent_regression_prepare.py`
- `scripts/es_co_apply_disclaimer_prefix_revision.py`
- `scripts/es_co_apply_first_revision.py`
- `scripts/es_co_apply_story006_c22_glossary_path.py`
- `scripts/es_co_apply_us_activation_to_co.py`
- `scripts/es_co_copy_date_revision.py`
- `scripts/es_co_create_korean_backtranslation.py`
- `scripts/es_co_finalize_drafts.py`
- `scripts/es_co_global_merge_260825.py`
- `scripts/es_co_local_adds_backtranslation.py`
- `scripts/es_co_local_adds_finalize.py`
- `scripts/es_co_rebuild_user_fixed_workbooks.py`
- `scripts/es_co_seed_us_glossary_highlights.py`
- `scripts/lowercase_029_child_account.py`
- `scripts/restore_029_us_highlights.py`
- `scripts/restore_045_us_highlights.py`
- `scripts/restore_trailing_whitespace_richtext.py`
- `scripts/restore_us_richtext.py`
- `scripts/workbook_add_target_sheet.py`
- `scripts/workbook_apply_sample_format.py`
- `scripts/workbook_fix_sheet_names.py`
- `scripts/workbook_harden_whitespace_batch.py`
- `scripts/workbook_mark_changed_text_red.py`
- `scripts/workbook_reset_text_colours.py`

직접 save 목록과 별개로 `es_co_regional_build_260825.py`, `build_korean_review_workbooks.py`,
`workbook_korean_review_draft.py`, `workbook_highlight_glossary.py`에는 계약 없는 guard 호출이
남아 있다. 이들은 각자 보존 검사가 있지만 공용 실행 계약 전체를 통과한 것으로 분류하지 않는다.

`batch_co_rollout.py`의 `__prep.xlsx`는 번역 입력용 중간 파일이며 최종 납품 파일이 아니다.
다만 직접 save→replace이고 계약 검증이 없다. add_target_sheet CLI만 전환하면 batch의 직접
저장은 그대로 남으므로 두 호출부를 같은 검증 저장 경로로 연결해야 한다.

copy_worksheet의 지원 범위도 확인했다. 메모리 내 합성 워크북으로 복제 시 차트 1→0,
데이터 유효성 1→0, 조건부 서식 1→0, freeze_panes C7→None을 재현했다. 따라서 “차트만
유실된다”는 것은 해당 샘플의 관측이며 일반 보존 보장은 아니다. 실제 outputs 12개에 그림이
없다는 리뷰의 관측은 이번 검토에서 재검증하지 않았으며 다른 객체의 부재를 뜻하지 않는다.

runner 연결만으로 복제 정확성이 해결되지는 않는다. 새 시트의 기대 구조/서식은 템플릿에서
독립적으로 계산하고, 지원하지 못하는 객체는 삭제 전에 명확히 거부하는 작은 변경이 우선이다.
특히 CO 유료 배치 전에는 이 준비 단계의 지원 객체 검사와 저장 검증을 완료해야 한다.

위험 우선순위는 앱 산출물의 납품 경계가 가장 높다. 실행 순서는 짧은 경계 문서화(이번 완료)
→ 파이프라인 candidate/검토/납품 연결 → target sheet 준비 경로 이전이 적절하다. CO 배치를
먼저 실행한다면 준비 경로 검증이 그 실행의 선행 조건이다. 나머지 일회용 writer는 사용 시
필요한 것부터 이전하고 전면 재작성은 하지 않는다. 이번 검토에서는 코드 변경이나 API 호출,
실제 workbook 저장을 하지 않았다.
