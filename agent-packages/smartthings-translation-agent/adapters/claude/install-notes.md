# Claude Code 설치 노트

> 상태: **문서만** (이번 단계는 Codex 우선). Claude 활성화 시 아래대로 복사.

## 설치 위치

- Project-local: `.claude/skills/smartthings-translation-agent/`
- User-global: `~/.claude/skills/smartthings-translation-agent/`

```bash
cp -r agent-packages/smartthings-translation-agent .claude/skills/
```

## 비고

- Claude Code는 `SKILL.md` + supporting files(`references/`, `scripts/`) 구조를 그대로 지원하므로, 공통 패키지를 변형 없이 복사하면 된다.
- `adapters/`는 복사할 필요 없음 (Codex 전용 메타데이터).
- 공통 `SKILL.md`에는 Claude 전용 frontmatter(`allowed-tools` 등)를 넣지 않았다. 필요하면 설치본의 `SKILL.md`에만 추가하고, 공통 본체는 건드리지 않는다.
- 현재 프로젝트 `.claude/`에는 `settings.local.json`만 있고 `skills/`는 없다 → 새로 생성됨.

## 동작 확인

```bash
# skill 디렉터리에서 스크립트가 src/ 를 찾는지
python .claude/skills/smartthings-translation-agent/scripts/rag_lookup.py \
  --query "turn on" --target-lang JA --json
```
자연어로는 "이 검수 결과 왜 Needs Revision이야?" 같은 요청으로 트리거 확인.
