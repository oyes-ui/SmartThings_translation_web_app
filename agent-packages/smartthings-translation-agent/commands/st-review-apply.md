---
description: 승인된 원어민 감수 판정을 반영해 최종 하이라이트 납품본 생성 (원본 불변, 크레딧 0)
argument-hint: <감수본.xlsx> <approval.json> --output <1차수용.xlsx> --glossary <Glossary.csv>
---

이 고급 명령은 `/st-apply`의 기존 실행 경로다. 사용자 기본 안내는 `commands/README.md`,
계약 실행·preview·재개와 실제 출력 경로는 `references/command-execution.md`를 따른다.
사용자 입력은 자연어로 받아 내부 manifest를 준비한다.

`/st-story-review`와 감수 의견 검토가 끝난 뒤에만 사용한다. 이 명령은 감수안을 새로 판단하지 않고, **사람이 승인한 manifest**의 `accept`·`partial`만 C열에 적용한다. 원본 감수본은 수정하지 않는다.

```bash
python agent-packages/smartthings-translation-agent/scripts/workbook_review_apply.py \
  <review_workbook.xlsx> <approval.json> \
  --output <story_1차수용_YYMMDD.xlsx> \
  --glossary <Glossary_260720.csv> \
  --drop-review-columns E:H \
  --app-root <SmartThings_app_repo> --json
```

## approval manifest

```json
{
  "protected_sheets": ["KR(한국)", "US(미국)", "CN(중국)", "BR(브라질)", "RU(러시아)"],
  "decisions": [
    {
      "sheet": "FR(프랑스)", "cell": "C8", "decision": "accept",
      "source_sheet": "US(미국)",
      "reason": "현지화 자연스러움 개선", "basis": "원문 의미·기능 조건 유지",
      "rag_basis": "RAG는 일관성 참고이며 거부 근거로 사용하지 않음"
    },
    {
      "sheet": "PT(포르투갈)", "cell": "C11", "decision": "partial",
      "source_sheet": "US(미국)", "final_value": "사람이 확정한 부분 반영 문안",
      "reason": "감수안 중 용어만 수용", "basis": "문법·용어집"
    },
    {"sheet": "TH(태국)", "cell": "C16", "decision": "hold", "reason": "UI 경로 리스크"}
  ]
}
```

## 판정 규칙

1. 현지화 개선은 기본 수용한다.
2. 원문 의미, 기능 조건, UI navigation path, glossary 강제 규칙, 문법을 해치면 `hold` 또는 `partial`로 둔다.
3. RAG는 기존 표현의 일관성 참고 자료다. exact match만으로 안전한 현지화 수정을 거부하지 않는다. RAG와 다른 결론이면 `rag_basis`에 이유를 기록한다.
4. 각 행에는 `현재 → 감수안 → 최종안 → 판정 → 쉬운 이유 → 원문/RAG 근거`가 Obsidian 리포트에 이미 있어야 한다.

## 완료 게이트

- `accept`·`partial` 결정과 실제 C열 값 diff가 정확히 일치한다.
- 보호 언어의 값·수식·병합 구조가 불변이다.
- 명시한 경우에만 E:H를 모든 시트에서 삭제한다. 삭제는 서식 used-range 축소와 구분한다.
- `C7:C28` 전체 시트를 Glossary 기준으로 재하이라이트하고, 하이라이트가 텍스트를 바꾸지 않았음을 확인한다.
- 최종 xlsx, 1차 수용본, highlight report, result manifest를 Obsidian 리포트에 경로와 함께 기록한다.
