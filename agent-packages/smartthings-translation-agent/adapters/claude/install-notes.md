# Claude Code 설치 노트

> 상태: **활성** — 이 repo의 `.claude/commands/`에 `/st-*` 명령이 설치돼 있다. 스킬 본체를 `.claude/skills/`로 복사하려면 아래 절차를 따른다.

## 설치 위치

- Project-local: `.claude/skills/smartthings-translation-agent/`
- User-global: `~/.claude/skills/smartthings-translation-agent/`

```bash
cp -r agent-packages/smartthings-translation-agent .claude/skills/
```

## 슬래시 명령어 (`/st-*`)

패키지의 `commands/*.md` 를 Claude Code 명령 디렉터리로 복사하면 `/st-help`, `/st-prompt`,
`/st-glossary` 등 14개 명령이 활성화된다.

```bash
# project-local
cp agent-packages/smartthings-translation-agent/commands/*.md .claude/commands/
# 또는 user-global
cp agent-packages/smartthings-translation-agent/commands/*.md ~/.claude/commands/
```

- 명령 파일은 작게 유지(스크립트 호출 + 참조문서 지시만) → 명령 1개를 실행할 때만 해당 파일이 로드된다.
- 명령은 `agent-packages/smartthings-translation-agent/scripts/...` 상대 경로로 스크립트를 호출하므로,
  repo 루트에서 Claude Code 를 실행하는 것을 전제로 한다. 설치 위치가 다르면 명령 파일의 경로를 조정한다.
- Codex/Antigravity 등 슬래시 명령이 없는 도구에서는 SKILL.md 의 `## 슬래시 명령어` 표로 동일 기능을 인식한다.

## 비고

- Claude Code는 `SKILL.md` + supporting files(`references/`, `scripts/`, `commands/`) 구조를 그대로 지원하므로, 공통 패키지를 변형 없이 복사하면 된다.
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
