---
description: 용어집 용어 rich text 하이라이트 (원본 불변, 크레딧 0)
argument-hint: <xlsx 경로> [--sheets "BR(브라질)"] [--include-source-sheets]
---

용어집 target term 글자 조각만 파란색 rich text 로 처리한다. 앱 highlight_only 파이프라인 호출, 원본 불변. **실행 전 delivery scope와 source/target 시트를 확인**하라. 셀 수정 후 납품본 생성에는 `/st-story-apply`를 우선 사용한다. 이 명령을 수동으로 쓸 때도 `--sheets`에 이번 납품 언어 전체를 넣고 `--include-source-sheets --cell-range C7:C28`를 사용한다.

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_highlight_glossary.py $ARGUMENTS --json
```
`*_highlighted_<ts>.xlsx` + `.highlight_report.txt`(불일치/괄호/대소문자) 생성. 참조: `agent-packages/smartthings-translation-agent/references/excel-workflow.md`
