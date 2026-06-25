---
description: 과거 번역 사례 RAG 조회 (offline=크레딧0 / semantic=임베딩)
argument-hint: --query <원문> --target-lang <코드> [--mode offline|semantic|auto]
---

과거 번역 사례를 조회한다. 키 없으면 offline(exact/keyword/메타, 크레딧 0), 키 있으면 semantic 자동.

```bash
python agent-packages/smartthings-translation-agent/scripts/rag_lookup.py $ARGUMENTS --json
```
예: `--query "Save energy" --target-lang JA` / `--keyword --query "節約" --target-lang JA` / `--story story_025 --target-lang JA`
참조: `agent-packages/smartthings-translation-agent/references/rag-workflow.md` (DB 데이터 품질 주의)
