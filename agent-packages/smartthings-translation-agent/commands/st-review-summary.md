---
description: 감수본·AI 검수 txt·리포트 요약 수치 산출 (읽기 전용, 크레딧 0)
argument-hint: --review-workbook <감수본.xlsx> [--fix-workbook <fix.xlsx>] [--ai-review-txt <review.txt>] [--report-md <report.md>]
---

원어민/번역사 감수본과 AI 검수 결과를 읽기 전용으로 대조해 최종 summary용 숫자를 산출한다. 원본 파일은 수정하지 않는다.

```bash
python agent-packages/smartthings-translation-agent/scripts/review_summary.py $ARGUMENTS --json
```

참조: `agent-packages/smartthings-translation-agent/references/response-patterns.md`
