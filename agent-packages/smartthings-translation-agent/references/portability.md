# 이식성 (Portability)

이 패키지는 **app-aware portable skill** 이다. standalone 이 아니라 SmartThings Translation Web App repo 를 조작하는 얇은 에이전트 계층이며, 단일 본체(`SKILL.md` + `references/` + `scripts/`)를 여러 에이전트 도구로 복사해 쓰도록 설계됐다. 도구별 메타데이터만 `adapters/`에 격리한다.

## 동작 레벨 (Level 1~4)

기능별로 app repo·키·의존성 요구가 다르다. `scripts/bootstrap.py` 가 현재 가능한 레벨을 자동 판정한다.

| Level | 모드 | 필요 조건 | 가능한 것 |
|-------|------|-----------|-----------|
| **1** | Excel-only | `openpyxl` 만 (app repo 불필요) | 워크북 분석/수정, section coherence 검토 |
| **2** | Offline RAG | app repo + `runtime/rag_db/rag_store.db` | + exact/keyword/메타 RAG (**API 키 불필요**, chromadb 미로드) |
| **3** | Semantic RAG | app repo + venv + `chroma/` + Gemini API key | + 임베딩 유사도 검색 |
| **4** | Full pipeline | Level 3 + glossary(csv/db) | + 용어집 검증·full app 연계 |

- 각 스크립트는 저장 config 또는 `--app-root` 로 app repo 를 해석한다. 자동 탐색은 repo-local 설치용 fallback 이다.
- Level 1~2 는 API 키 없이 동작한다. Level 3~4 만 키가 필요하다.
- 다른 skill 에 Python 의존성 해결을 위임하지 않는다. 의존성은 `requirements-excel.txt`/`requirements-rag.txt` 또는 app `requirements.txt` 로 명시 설치한다.

## 설계 원칙

- `SKILL.md` frontmatter는 공통 필드(`name`, `description`)만 사용. 특정 도구 전용 문법(`allowed-tools`, `context: fork` 등)은 본체에 넣지 않는다.
- `scripts/`는 에이전트 런타임에 의존하지 않는 **일반 Python CLI**. 어느 도구에서 호출해도 동일하게 동작하며 `--json`으로 파싱 가능.
- app_root 는 `bootstrap.py` 가 `--app-root` → `SMARTTHINGS_APP_ROOT` → config → cwd/파일위치 탐색 순으로 해석한다. skill 실행 세션에서는 `--app-root` 또는 config 를 우선 사용하고, 절대 경로를 패키지에 하드코딩하지 않는다.
- `.env`, API 키, 업로드 원본은 패키지에 **포함하지 않는다**.
- 저장 config 는 `~/.smartthings_translation_agent/config.json`(app_root 만). skill 패키지 외부에 둔다.

## 도구별 설치 위치

| 도구 | 설치 경로 | 비고 |
|------|-----------|------|
| **Codex** | repo-local `skills/smartthings-translation-agent/` 또는 `$CODEX_HOME/skills/` | `adapters/codex/agents/openai.yaml` 포함. → `adapters/claude/install-notes.md` 형식의 `adapters/codex` 참고 |
| **Claude Code** | `.claude/skills/smartthings-translation-agent/` (project-local) 또는 `~/.claude/skills/` | `adapters/claude/install-notes.md` 참조 |
| **Antigravity** | `.agents/skills/smartthings-translation-agent/` | `adapters/antigravity/install-notes.md` 참조. ⚠ 이 프로젝트의 `.agent/`(단수)는 app orchestration 런타임이므로 사용 금지 |

## ⚠ 이 프로젝트 고유 주의

- `.agent/`(단수)는 **app orchestration 런타임 디렉터리**(`.tasks/`, `runs/`, `state.json`). skill 설치 경로로 절대 쓰지 않는다.
- Antigravity 기본 경로는 `.agents/`(복수)이므로 충돌하지 않는다.
- 이 skill 자체는 app orchestration 런타임이 **불필요**하다. 스크립트는 독립 실행된다.

## 의존성 매트릭스 (공유 시 핵심)

기능별로 필요한 Python 패키지가 다르다. **Excel 기능은 RAG 스택 없이 동작**하도록 분리돼 있다.

| 스크립트 / 모드 | 필요 패키지 | API 키 | 설치 크기 | src/ 필요? |
|----------|-------------|--------|-----------|------------|
| `bootstrap.py` | (표준 라이브러리만) | ❌ | 0 | ❌ |
| `workbook_inspect.py` | `openpyxl` | ❌ | 가벼움 | ❌ (상수 fallback 내장) |
| `workbook_apply_edits.py` | `openpyxl` | ❌ | 가벼움 | ❌ |
| `rag_lookup.py` **offline** | (표준 `sqlite3`만) | ❌ | 0 | ❌ (DB 파일만 있으면 됨) |
| `rag_lookup.py` **semantic** | `chromadb` + `google-genai` + `python-dotenv` | ✅ Gemini | ~92M | ✅ 필수 |

`rag_lookup.py`는 offline mode일 때 chromadb/google-genai를 **로드하지 않고** 표준 라이브러리 `sqlite3`만 쓴다 → 키·무거운 의존성 없이 exact/keyword/메타데이터 조회 가능. semantic(유사도)만 키+RAG 스택이 필요하다.

```bash
# Excel 기능만 → 최소 설치
pip install -r scripts/requirements-excel.txt

# RAG 조회까지 → 전체 설치 (버전 핀 적용됨)
pip install -r scripts/requirements-rag.txt
```

**⚠ 버전 핀 주의**: `chromadb`(현재 1.5.x), `google-genai`(1.73.x)는 메이저 API 변경이 잦다. `requirements-rag.txt`의 상한(`<2`)을 풀고 최신을 설치하면 `get_chroma_collections()` / `embed_content()` 시그니처 불일치로 깨질 수 있다. 받는 사람에게 **이 requirements 파일로 설치**하도록 안내한다.

**디커플링 동작**: `workbook_inspect.py`는 `translation_web_app.rag_db_builder`에서 상수를 재사용하되, 그 모듈(또는 chromadb)이 없으면 로컬 fallback 상수로 동작한다. 즉 **엑셀만 보려는 사람은 `openpyxl`만 설치**하면 되고, src/ 나 RAG 스택이 없어도 분석이 된다.

## 데이터·시크릿은 패키지에 없다 (gitignore)

코드(이 패키지 + `src/`)는 git으로 공유되지만, 아래는 `.gitignore`로 **제외**된다. 별도 전달 필요:

| 항목 | 경로 | 공유 방법 |
|------|------|-----------|
| API 키 | `.env` (`GEMINI_API_KEY`) | 받는 사람이 직접 발급·입력 |
| RAG DB | `runtime/rag_db/` (~33M) | zip 별도 전달, 또는 받는 사람이 `rag_db_builder --build-all` 재빌드(크레딧 소모) |
| 용어집 | `runtime/glossary/latest_glossary.csv` | zip 별도 전달 (최초 조회 시 DB 자동 시드) |
| 엑셀 원본 | `@translation_data/@excel/*.xlsx` (~2.9M) | zip 별도 전달 |

### 받는 사람 셋업 체크리스트
1. `pip install -r scripts/requirements-rag.txt` (또는 `-excel`)
2. `.env`에 본인 `GEMINI_API_KEY` 입력
3. 엑셀/용어집/RAG DB 압축 풀어 위 경로에 배치 (재빌드할 거면 엑셀만)
4. (재빌드 시) `PYTHONPATH=src python -m translation_web_app.rag_db_builder --build-all`

## 설치 방법 (공통)

```bash
# 예: Claude Code project-local 설치
cp -r agent-packages/smartthings-translation-agent .claude/skills/

# 예: Codex repo-local (어댑터 포함)
cp -r agent-packages/smartthings-translation-agent skills/
cp agent-packages/smartthings-translation-agent/adapters/codex/agents/openai.yaml \
   skills/smartthings-translation-agent/agents/openai.yaml
```

설치 후 skill 폴더에서 아래처럼 app repo 를 한 번 연결한다.

```bash
cd <installed-skill-folder>
python scripts/bootstrap.py --app-root /path/to/SmartThings_translation_web_app --save
```

repo 밖에 설치할 경우 저장된 `app_root` 또는 매 호출의 `--app-root` 가 원본 repo를 가리켜야 한다.

## fallback 포인터 (선택)

루트 `AGENTS.md`에 아래 한 줄을 추가하면, skill discovery를 지원하지 않는 에이전트도 패키지를 찾을 수 있다:

```markdown
## Translation Agent Skill
번역·검수 보조 에이전트 패키지는 `agent-packages/smartthings-translation-agent/SKILL.md` 참조.
```
