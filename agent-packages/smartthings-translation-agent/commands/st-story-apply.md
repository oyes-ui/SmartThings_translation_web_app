---
description: 승인된 story 수정안을 납품용 하이라이트 복사본으로 생성 (원본 불변, 크레딧 0)
argument-hint: <xlsx> <edits.json> --delivery-sheets "VN(베트남),TH(태국)" --glossary <Glossary.csv>
---

이 고급 명령은 `/st-apply`의 기존 실행 경로다. 사용자 기본 안내는 `commands/README.md`,
계약 실행·preview·재개와 실제 출력 경로는 `references/command-execution.md`를 따른다.
사용자 입력은 자연어로 받아 내부 manifest를 준비한다.

`/st-story-review`에서 확정·승인된 수정안만 반영한다. 원본은 수정하지 않으며, **이번 납품 범위 전체**를 재하이라이트한 파일만 최종본으로 안내한다.

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_story_apply.py \
  <workbook.xlsx> <edits.json> \
  --delivery-sheets "VN(베트남),TH(태국),ID(인도네시아)" \
  --glossary <Glossary_049_260715.csv> \
  --app-root <SmartThings_app_repo> --json
```

## 완료 조건

1. staging에서 수정본을 만들고 최종본까지 검증한 뒤 버전 배치로 공개한다.
2. 승인된 셀만 바꾼다.
3. `--delivery-sheets`와 필요한 KR/US source 시트의 `C7:C28` rich text 하이라이트를 glossary 기준으로 재생성한다.
4. 원본 대비 값 변경이 승인된 셀과 정확히 일치하고, highlight report에 delivery scope의 모든 시트가 실제 처리됐는지 검증한다.
5. 최종 `.xlsx`, highlight report, `.delivery.json` manifest 경로를 출력한다.

## 규칙

- `--delivery-sheets`는 필수다. 워크북에 있어도 이번 납품에서 제외된 시트는 넣지 않는다.
- `st-edit`는 저수준 편집용이다. 실제 납품본은 반드시 이 명령의 `final` 산출물을 사용한다.
- Obsidian에는 검수 직후 제안만 기록하고, 이 명령이 만든 `.delivery.json`을 근거로 `/st-obsidian-report`에서 반영 상태와 최종본 경로를 갱신한다.
