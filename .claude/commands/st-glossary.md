---
description: 용어집 조회/검색/CRUD/CSV import·export (크레딧 0)
argument-hint: <status|locales|list|add|update|delete|import|export> [옵션]
---

앱 GlossaryStore 래퍼로 용어집을 다룬다(크레딧 0).

```bash
python agent-packages/smartthings-translation-agent/scripts/glossary_manage.py $ARGUMENTS --json
```
읽기: `list --search AI` / `status` / `export --out /tmp/g.csv` → 바로 실행.
쓰기(add/update/delete/import): **--apply 가드가 강제**된다. 먼저 --apply 없이 실행해 변경 미리보기를
사용자에게 보여주고 **승인받은 뒤** --apply 를 붙여 재실행하라. delete·`import --mode replace` 는 비가역.
