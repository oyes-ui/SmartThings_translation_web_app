# Antigravity 설치 노트

> 상태: **문서만** (이번 단계는 Codex 우선). Antigravity 활성화 시 아래대로 복사.

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

## 동작 확인

```bash
python .agents/skills/smartthings-translation-agent/scripts/workbook_inspect.py \
  <story.xlsx> --json
```
