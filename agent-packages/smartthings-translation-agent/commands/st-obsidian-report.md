# /st-obsidian-report

SmartThings 번역/검수 작업 내용을 Obsidian용 Markdown 리포트로 새로 작성하거나 기존 리포트의 언어별 섹션을 증분 갱신한다.

## Arguments

`$ARGUMENTS`

권장 입력:

- 리포트 주제(예: `049 Quick Panel BR/RU/CN 용어집 필터`)
- workbook 경로
- 대상 언어
- Obsidian 리포트 저장 경로
- 포함할 판단/메모

## Workflow

1. `references/glossary-report-workflow.md`의 "Obsidian 리포트 작성" 섹션을 따른다.
2. 새 리포트면 Markdown front matter를 포함해 초안을 만들고, 기존 리포트면 해당 언어 섹션만 갱신한다.
3. 언어별 섹션에 `AI 판정`, `최종 판단`, `source group`, `RAG 근거`, `최종 문안/제안`, `반영 상태`를 기록한다.
4. 기본 운영은 검수 직후 제안·근거만 기록하고 `반영 상태: 미확인`으로 둔다. 여러 언어 검수 후 현재 납품 워크북을 한 번 읽어 반영 상태만 일괄 갱신한다.
5. `/st-story-apply`를 실행한 경우 `.delivery.json` manifest를 읽어 `최종본 경로`, `delivery scope`, `glossary`, `값 변경 검증`, `highlight report`를 작업 메모에 기록한다. manifest가 없는 수정본은 `반영 완료`로 표시하지 않는다.
6. Obsidian vault/iCloud 경로는 먼저 `ls -la`로 접근 가능 여부를 확인한다.
7. workspace 밖 경로에 저장해야 하면 필요한 권한 승인을 받은 뒤 복사한다.
8. 저장 후 `ls -l`로 파일 존재를 확인한다.

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
