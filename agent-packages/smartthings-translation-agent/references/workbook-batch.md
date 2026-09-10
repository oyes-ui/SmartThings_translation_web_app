# 범용 작업 설정과 재개

업무 안내 정본은 [업무 가이드](workflow-guide.md)입니다. 아래 JSON과 명령은 에이전트용 내부 인터페이스입니다. 사용자에게 작성을 요구하지 않습니다. 새로운 slash command는 없습니다.

## 설정 구성

에이전트가 파일과 언어마다 job을 구성하고 용어집 적용·제외안을 사용자에게 제시합니다. 확정되면 `glossary_approved: true`로 기록합니다. 기본 원문 그룹은 유지되며 `source_sheet`는 예외가 있을 때만 지정해도 됩니다. `sheet_langs`는 비표준 시트의 실제 앱 언어명·용어집 컬럼 code를 지정합니다.

```json
{
  "app_root": "/absolute/app",
  "obsidian_dir": "/absolute/vault/Reviews",
  "jobs": [{
    "workbook": "/absolute/story_001.xlsx",
    "story": "001",
    "sheet": "DE(독일)",
    "glossary": "/absolute/glossary.csv",
    "activation_manifest": "/absolute/approved-activation.json",
    "glossary_approved": true,
    "translate": true,
    "review": true
  }]
}
```

- `activation_manifest`는 적용·제외 예외가 없으면 생략합니다. 그 경우에도 기본 적용안의 사용자 확정을 기록합니다.
- 새 시트가 필요할 때만 `prepare: {"template_sheet": "ES(스페인)", "lang_code": "es_CO"}`처럼 지정합니다. 지역명은 예시이며 모든 시트명·언어코드는 설정입니다. 복제 불가 객체가 있으면 명시적인 별도 양식 작업으로 안내합니다.
- `output_dir`을 지정하면 그 폴더 아래 `<job_id>/<원본파일명>`으로 초벌을 저장합니다. 생략하면 작업 폴더의 `draft/`를 사용하며, 다른 내용의 기존 파일은 덮어쓰지 않습니다.
- `translate: false`는 기존 번역본 검수, `review: false`는 초벌까지만 진행, `prep_only: true`는 워크북 준비까지만 진행합니다.
- 기본 `api_audit`는 false입니다. 명시 요청한 API 검수에만 true를 지정합니다. 역번역도 요청한 경우에만 `with_backtranslation: true` 또는 `backtranslation_sheet`·`backtranslation_lang`을 지정합니다. 역번역 시트가 필요하면 준비 단계에서 생성합니다.
- 실행 중 새 기본 용어집을 자동으로 반영하지 않습니다. 입력 워크북·CSV·활성화 파일·언어 매핑을 작업 폴더에 복사하고 해시를 확인합니다. 같은 파일·언어의 중복 job은 거부합니다.
- 다중 파일은 각 파일·언어를 독립 job으로 처리합니다. 언어별 초벌 사본을 만들며 원본에 여러 job이 동시에 쓰지 않습니다. 자동으로 모든 언어를 한 Excel에 재조립하는 단계는 없습니다.

## 실행

```bash
python scripts/workbook_batch.py prepare --config <config.json> --work-dir <work-dir>
python scripts/workbook_batch.py advance --manifest <work-dir>/workflow.json --pipeline
python scripts/workbook_batch.py status --manifest <work-dir>/workflow.json
```

`--pipeline`은 해당 범위의 유료 초벌을 이미 승인한 경우에만 붙입니다. 없으면 유료 호출 직전에 `awaiting_api_approval`로 대기합니다. `status`는 읽기 전용입니다.

초벌 이후 활성 에이전트는 각 job의 `ready_for_agent` prompt를 읽고 결과를 같은 폴더에 `cell_review.json` → `sheet_review.json` → `lead_review.json` 순으로 저장합니다. 각 단계 뒤 다음 명령을 실행하고 다음 prompt를 처리합니다.

```bash
python scripts/workbook_batch.py advance --manifest <work-dir>/workflow.json
```

이 과정을 완료·실패 또는 사용자 판단이 필요한 지점까지 이어갑니다. 기존 [검수 규약](review-workflow.md)의 셀·시트·리드 역할을 지킵니다. 조정기는 검수용 API·새 데몬을 호출하지 않습니다. 이미 완료된 API 초벌은 재실행하지 않습니다. 검수 단계 JSON을 교정한 후 같은 advance로 무료 재개할 수 있습니다.

유료 호출 중 중단되어 결과가 불확실하면 `api_recovery_required`, 호출 실패면 `translation_error`입니다. 실제 API 산출물을 확인한 후 재시도가 필요하고 사용자가 비용을 승인했을 때만 `--pipeline --retry-job <job_id>`를 지정합니다. 저장된 완료 receipt가 있으면 API 없이 로컬 처리를 재개합니다. 성공 job은 다른 job 실패와 무관하게 계속 진행합니다.

## 산출물과 승인

- `jobs/<id>/settings.json`: 고정 설정, 원래 입력 경로와 스냅샷 해시.
- `jobs/<id>/draft/`: 초벌 사본. 최종 납품 계약이 아닙니다.
- `jobs/<id>/review/`: 기존 단계별 prompt·검수 결과·상세 manifest.
- `reports/review-<workflow-id>-<id>.md`: 파일·언어별 상세 리포트.
- `reports/approval-<workflow-id>.md`와 `.json`: Story·언어별 통합 승인검토표와 원본 후보 색인. MD 표시를 위해 개행·파이프를 이스케이프하지만 JSON의 문안·후보 ID는 원본과 같습니다.
- 지정 Obsidian 폴더: 같은 MD를 저장. 자동 생성 블록 밖의 사용자 메모 보존. 폴더 미지정은 `location_required`, 기존 비관리 문서 충돌은 `blocked_existing_note`.

사용자 승인 내용을 원본 검수 manifest에 반영하고 기존 `/st-apply`를 수행합니다. `--workflow-settings <job>/settings.json`과 그 안의 `--glossary` 경로를 넘기면 원문·언어 매핑·활성화 설정을 같은 값으로 재사용합니다. 다른 언어 job의 후보를 섞지 않습니다. 승인표 색인을 그대로 apply 입력으로 사용하지 않습니다.

## 호환성과 경계

`batch_co_rollout.py`는 기존 지역 기본값을 범용 job 설정으로 바꾸는 호환 래퍼입니다. 신규 실행은 `--glossary-approved`, 유료 호출은 `--pipeline`이 필요합니다. 과거 `manifest.json`의 실패를 자동 재실행하지 않으며, 이전 완료 산출물부터 새 작업을 시작할 수 있습니다. 준비 전용 job을 번역 작업으로 변경하려면 그 준비 산출물로 새 설정을 구성합니다.

초벌 workbook 저장은 외부 앱에서 수행합니다. 이 연결은 최종 납품용 guard를 앱 전체에 이식하지 않습니다. 워크북 준비·승인 반영은 패키지의 검증 저장 경로를 사용합니다. 임의 구조 변환·일회용 레거시 writer 전체 이전은 이번 범위가 아닙니다.
