# SmartThings 번역 에이전트 Portable Skill 패키지 계획

## 요약

- 기존 계획을 "Codex 전용 skill"이 아니라 이식 가능한 Agent Skill 패키지와 도구별 어댑터 구조로 확장한다.
- 공통 본체는 Agent Skills 스타일의 `SKILL.md`, `references/`, `scripts/`만 사용하고, Codex/Claude/Antigravity 전용 파일은 별도 `adapters/`에 둔다.
- Claude Code는 `.claude/skills/<name>/SKILL.md` 구조를 지원하고, Antigravity는 현재 문서 기준 `.agents/skills`를 기본 경로로 사용하므로, 설치 대상만 다르게 복사할 수 있게 만든다.

## 패키지 구조

```text
agent-packages/
└── smartthings-translation-agent/
    ├── SKILL.md
    ├── references/
    │   ├── rules-sources.md
    │   ├── rag-workflow.md
    │   ├── excel-workflow.md
    │   ├── response-patterns.md
    │   └── portability.md
    ├── scripts/
    │   ├── rag_lookup.py
    │   ├── workbook_inspect.py
    │   └── workbook_apply_edits.py
    └── adapters/
        ├── codex/
        │   └── agents/openai.yaml
        ├── claude/
        │   └── install-notes.md
        └── antigravity/
            └── install-notes.md
```

- `agent-packages/.../SKILL.md`를 단일 원본으로 유지한다.
- Codex용 `agents/openai.yaml`은 공통 루트가 아니라 `adapters/codex/`에 둔다. 이 파일은 UI 메타데이터일 뿐 실행 모델과 무관함을 명시한다.
- Claude/Antigravity용 별도 `SKILL.md`를 만들지 않는다. 중복본이 생기면 규칙 불일치가 생기기 쉽다.
- 설치 시 각 도구의 skill 디렉터리로 공통 패키지를 복사한다.

## 호환성 원칙

- `SKILL.md` frontmatter는 최대한 공통 필드만 사용한다.
  - `name`
  - `description`
- Claude 전용 기능인 `allowed-tools`, `context: fork`, 동적 명령 삽입 문법은 공통 `SKILL.md`에 넣지 않는다.
- Antigravity 전용 설정이나 Codex UI 메타데이터는 `adapters/`에 격리한다.
- 모든 스크립트는 에이전트 런타임에 의존하지 않고 일반 CLI로 실행 가능해야 한다.
- 경로는 가능한 한 `PROJECT_ROOT` 자동 탐색 방식으로 처리하고, 절대 경로는 reference 문서에만 보조 정보로 둔다.
- `.env`, `.agent/.env`, API 키, 업로드 원본 파일은 패키지에 포함하지 않는다.

## 설치 대상

- Codex repo-local:
  - `skills/smartthings-translation-agent/`
  - 필요 시 `adapters/codex/agents/openai.yaml`을 `skills/.../agents/openai.yaml`로 포함한다.
- Claude Code project-local:
  - `.claude/skills/smartthings-translation-agent/`
  - Claude 문서 기준 skill은 `SKILL.md`와 supporting files를 포함할 수 있으므로 공통 패키지를 그대로 복사한다.
- Antigravity project-local:
  - `.agents/skills/smartthings-translation-agent/`
  - `.agent/skills`는 기존 프로젝트의 app orchestration 런타임 디렉터리와 충돌 위험이 있으므로 사용하지 않는다.
- 범용 fallback:
  - 루트 `AGENTS.md`에 "이 프로젝트의 번역 에이전트 패키지는 `agent-packages/smartthings-translation-agent/`를 읽으라"는 짧은 포인터를 추가하는 방식을 별도 옵션으로 둔다.

## 스크립트 설계

- `rag_lookup.py`
  - `.env` 로딩 포함.
  - `src/` import 경로 자동 설정.
  - `Japanese`와 `JA(일본)` 입력을 모두 허용하고 내부에서 DB 저장 형식인 시트명으로 정규화한다.
- `workbook_inspect.py`
  - `rag_db_builder.py`의 `STORY_ID_CELL`, `CONTENT_ROW_START`, `CONTENT_ROW_END`, `SECTION_COL`, `CONTENT_COL`을 import한다.
  - SmartThings Excel 구조를 분석하되 파일은 수정하지 않는다.
- `workbook_apply_edits.py`
  - 승인된 edits JSON만 처리한다.
  - 원본 불변, 타임스탬프 수정본 생성, `.tmp` 저장 후 `os.replace()` atomic write를 지킨다.
- 모든 스크립트는 `--json` 출력 옵션을 제공해 Claude, Codex, Antigravity 어느 쪽에서도 파싱하기 쉽게 한다.

## 이동성 보강

- `references/portability.md`를 추가해 도구별 설치 위치와 제한 사항을 정리한다.
- 패키지 내부 문서는 "Codex가 실행한다" 같은 표현 대신 "agent" 또는 "사용 중인 에이전트"로 쓴다.
- `scripts/`는 Python 표준 CLI로 유지하고, 에이전트별 tool 호출 문법을 문서에 넣지 않는다.
- Obsidian 경로는 로컬 전용이므로 필수 의존성으로 만들지 않는다. 없으면 `docs/comprehensive_rules.md`만으로 작동한다.
- `.agent/`는 현재 프로젝트 app orchestration 런타임 영역이므로 portable skill 설치 경로로 사용하지 않는다.

## 검증 계획

- 공통 패키지 구조 확인:
  - `find agent-packages/smartthings-translation-agent -type f | sort`
- Codex 검증:
  - repo-local `skills/`로 복사 후 skill이 의도한 설명/스크립트/reference를 참조하는지 확인한다.
- Claude 검증:
  - `.claude/skills/smartthings-translation-agent/`로 복사 후 `/smartthings-translation-agent` 또는 관련 자연어 요청으로 동작을 확인한다.
- Antigravity 검증:
  - `.agents/skills/smartthings-translation-agent/`로 복사 후 skill discovery 여부와 스크립트 실행 가능 여부를 확인한다.
- 공통 기능 검증:
  - RAG 조회, Excel inspect, Excel apply 테스트는 어느 에이전트에서 실행해도 동일한 CLI 출력이 나와야 한다.

## 참고 근거

- Claude Code 문서는 `SKILL.md` 기반 skill, supporting files, `.claude/skills/<name>/SKILL.md` 위치를 지원한다고 설명한다: https://docs.claude.com/en/docs/claude-code/skills
- Google Antigravity 문서는 skill이 `SKILL.md`를 포함한 폴더이며, 현재 기본 위치가 `.agents/skills`이고 `.agent/skills`는 backward compatibility로 유지된다고 설명한다: https://antigravity.google/docs/skills
- 이 프로젝트에서는 `.agent/`가 app orchestration 런타임 디렉터리이므로 Antigravity용 설치는 `.agents/skills`를 우선한다.
