# /st-obsidian-report

SmartThings 번역/검수 작업 내용을 Obsidian용 Markdown 리포트로 준비·검색·명시 발행·납품 상태 갱신한다.

## Arguments

`$ARGUMENTS`

권장 입력:

- 리포트 주제(예: `049 Quick Panel BR/RU/CN 용어집 필터`)
- workbook 경로
- 대상 언어
- Obsidian 리포트 저장 경로
- 포함할 판단/메모

## Workflow

1. `references/glossary-report-workflow.md`와 `docs/obsidian-skill-dependencies.md`를 따른다.
2. 기본은 `stage`: 표준 Markdown 리포트를 workspace의 Obsidian 초안으로 변환한다. 구조화 finding, anchor, `before/after`, rule ID, approval 상태는 유지한다.
3. `search`는 사용자가 Obsidian 자료 검색·비교를 **명시**했을 때만 실행한다. `--vault-name`과 실행 중인 Obsidian CLI가 있으면 CLI를 사용하고, 아니면 읽기 전용 파일 검색으로 fallback한다.
4. renderer는 명시한다: `approval_review`는 Story별 `[!example]-` 접이식 callout과 셀별 source/current/proposed/decision/rule IDs/bracket reasons/RAG advisory를 사용하고, `full_audit`는 앱/에이전트 raw payload를 보존한다. HTML `<details>`는 금지한다.
5. 발행 전에는 Markdown 원문에서 `<details>`가 없는지, 모든 finding에 rule IDs·approval 상태가 있는지, 접이식 callout과 표가 깨지지 않는지 확인한다.
4. `publish`와 `init-base`는 vault를 변경하므로 사용자 승인과 `--apply`가 필수다. 기존 리포트는 `--locale`로 지정한 언어 섹션만 증분 갱신한다.
5. `sync-status`는 `/st-story-apply`의 `.delivery.json` 또는 `/st-review-apply`의 `.review_apply.json`처럼 유효한 result manifest만 근거로 `applied`와 납품 검증을 기록한다.

```bash
# workspace 초안 생성 (기본, vault 미변경)
python scripts/obsidian_workflow.py stage outputs/review/review-001.md \
  --output outputs/obsidian/review-001.md

# 명시 요청한 vault 검색 (CLI 미가동이면 파일 검색 fallback)
python scripts/obsidian_workflow.py search "/path/to/vault" "콜롬비아 tú" --limit 10

# vault 발행 / 기존 CO 섹션만 갱신 (사용자 승인 후에만)
python scripts/obsidian_workflow.py publish outputs/obsidian/review-001.md "/path/to/vault" \
  "SmartThings/review-001.md" --locale CO --apply

# front matter 기반 읽기 전용 현황판 생성 (사용자 승인 후에만)
python scripts/obsidian_workflow.py init-base "/path/to/vault" \
  --output "SmartThings/SmartThings Translation Reviews.base" --apply
```

## Report Shape

기본 섹션:

- 목적
- 결론
- source group 확정문구 기준 실제 매칭 또는 검수 기준
- 공통 판단
- 언어별 검수 섹션
- 다음 작업
- 작업 메모

## Rules

- 리포트 저장은 Markdown `.md`로 한다.
- Excel 원본은 수정하지 않는다.
- 리포트에 API 키, `.env` 내용, 비밀 값은 쓰지 않는다.
- 사용자가 경로를 지정하면 그 경로를 우선하고, 기존 개인 경로를 다른 환경에 일반화하지 않는다.
- Obsidian 노트는 canonical 규칙·glossary보다 낮은 보조 근거다. 사용자가 명시하지 않으면 자동 검색하지 않는다.
