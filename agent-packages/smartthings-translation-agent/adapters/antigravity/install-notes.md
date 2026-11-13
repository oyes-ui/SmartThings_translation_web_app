# Antigravity 설치 노트

> 공통 패키지를 연결하는 안내다. 현재 환경의 설치 여부는 별도로 확인한다.
> 기본 명령과 지원 범위는 [명령 안내](../../commands/README.md)·[업무 가이드](../../references/workflow-guide.md)를 따른다.

## 설치 위치

- Project-local: `.agents/skills/smartthings-translation-agent/` (복수 `.agents`)

```bash
cp -r agent-packages/smartthings-translation-agent .agents/skills/
```

## ⚠ 중요: `.agent/` vs `.agents/`

- 이 프로젝트의 **`.agent/`(단수)** 는 app orchestration 런타임 디렉터리다(`.tasks/`, `runs/`, `state.json`, `heartbeat.json`). **여기에 skill을 설치하면 런타임 파일과 충돌**한다.
- Antigravity 기본 skill 경로는 **`.agents/`(복수)** 이므로 단수 `.agent/`와 구분된다. 반드시 복수형 `.agents/skills/`에 설치한다.
- Antigravity 문서가 backward-compat로 `.agent/skills`를 언급하더라도, 이 프로젝트에서는 단수 `.agent/`가 app orchestration 영역이므로 **사용하지 않는다**.

## 비고

- Antigravity도 `SKILL.md` 기반 폴더 구조를 지원하므로 공통 패키지를 그대로 복사.
- `adapters/`는 복사 불필요.
- 패키지 문서는 "Codex가 실행한다" 같은 도구 종속 표현을 피하고 "에이전트"로 통일해, 어느 도구에서 읽어도 자연스럽게 동작하도록 작성됨.

## 사용자 명령과 실행 루트

기본은 `/st-start`, `/st-ask`, `/st-translate`, `/st-inspect`, `/st-edit`, `/st-apply`다.
고급·호환 명령은 [공통 목록](../../commands/README.md)에서 필요할 때 확인한다.
명령 진입점도 패키지의 `commands/*.md`로 위임하며 절차를 복제하지 않는다.
명령 검색을 사용할 수 없는 환경에서는 같은 요청을 자연어로 전달한다.

설치된 패키지 폴더를 실행 루트로 사용하고 앱 저장소를 연결한다.

```bash
cd <installed-skill-folder>
python scripts/bootstrap.py --app-root /path/to/SmartThings_translation_web_app --json
python scripts/bootstrap.py --app-root /path/to/SmartThings_translation_web_app --save
```

## 동작 확인

```bash
python scripts/workbook_inspect.py \
  <story.xlsx> --json
```
