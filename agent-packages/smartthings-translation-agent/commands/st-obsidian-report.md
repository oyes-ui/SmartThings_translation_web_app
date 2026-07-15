# /st-obsidian-report

SmartThings 번역/검수 작업 내용을 Obsidian용 Markdown 리포트로 정리하고, 사용자가 지정한 vault 경로에 저장한다.

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
2. Markdown front matter를 포함해 초안 리포트를 만든다.
3. 차후 확장 가능하도록 언어별 섹션을 분리한다.
4. Obsidian vault/iCloud 경로는 먼저 `ls -la`로 접근 가능 여부를 확인한다.
5. workspace 밖 경로에 저장해야 하면 필요한 권한 승인을 받은 뒤 복사한다.
6. 저장 후 `ls -l`로 파일 존재를 확인한다.

## Report Shape

기본 섹션:

- 목적
- 결론
- US 확정문구 기준 실제 매칭 또는 검수 기준
- 공통 판단
- 언어별 검수 섹션
- 다음 작업
- 작업 메모

## Rules

- 리포트 저장은 Markdown `.md`로 한다.
- Excel 원본은 수정하지 않는다.
- 리포트에 API 키, `.env` 내용, 비밀 값은 쓰지 않는다.
- 사용자가 경로를 지정하면 그 경로를 우선하고, 기존 개인 경로를 다른 환경에 일반화하지 않는다.

