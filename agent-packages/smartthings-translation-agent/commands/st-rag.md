---
description: 과거 번역 사례 및 표현 일관성 RAG 조회 (offline=크레딧0 / semantic=임베딩)
argument-hint: --query <원문> --target-lang <코드> [--source-lang English|Korean] [--story ...] [--section ...]
---

과거 번역 사례를 조회한다. 키 없으면 offline(exact/keyword/메타, 크레딧 0), 키 있으면 semantic 자동.

```bash
python agent-packages/smartthings-translation-agent/scripts/rag_lookup.py $ARGUMENTS --json
```
표현 통일 조회에는 대상 언어, source group, 콘텐츠 유형을 함께 밝힌다. description 선택에는 description 사례를 우선하고, disclaimer/navigation 사례를 일반 문구의 1차 근거로 쓰지 않는다.

예: `--query "Save energy" --target-lang JA --source-lang Korean` / `--keyword --query "節約" --target-lang JA --source-lang Korean` / `--story story_025 --target-lang JA`

응답에는 exact/keyword/semantic 유형, 콘텐츠 유형(title/description/disclaimer), 사례 부족 여부를 반드시 표시한다. 사례가 없으면 "RAG 근거 없음: 규칙/BX 기준의 권장"이라고 밝힌다.
참조: `agent-packages/smartthings-translation-agent/references/rag-workflow.md` (DB 데이터 품질 주의)
