---
name: "source-command-st-ragdb"
description: "RAG DB 현황/빌드/업데이트 (status=0, build=임베딩 크레딧)"
---

# source-command-st-ragdb

Use this skill when the user asks to run the migrated source command `st-ragdb`.

## Command Template

앱 rag_db_builder CLI 로 RAG 인덱스를 관리한다. **status 외 빌드/업데이트는 임베딩 크레딧을 소모**하므로 실행 전 확인.

```bash
PYTHONPATH=src python -m translation_web_app.rag_db_builder --status   # 크레딧 0
# 빌드(승인 후): --pilot | --build-all [--force] | --update-story <id>
```
app repo 루트에서 실행. 참조: `agent-packages/smartthings-translation-agent/references/rag-workflow.md`
