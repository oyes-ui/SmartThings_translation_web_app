# /st-rules

SmartThings 번역·검수 규칙을 canonical source 기준으로 답한다.

## Arguments

`$ARGUMENTS`

권장 입력:

- 질문할 규칙 또는 언어
- 필요하면 관련 셀/문구

## Workflow

1. `references/rules-sources.md`를 읽고 우선순위를 확인한다.
2. 가능한 경우 app repo의 `src/translation_web_app/prompt_modules.py`를 최종 기준으로 확인한다.
3. 보조 설명이 필요하면 `docs/comprehensive_rules.md`를 참조한다.
4. 답변은 규칙 위반, 스타일 선호, 예외를 구분한다.

## Rules

- 기억이나 일반 번역 관행만으로 단정하지 않는다.
- 충돌 시 런타임 프롬프트 소스인 `prompt_modules.py`를 우선한다.
- 사용자가 Obsidian 노트를 명시하지 않으면 개인 노트는 자동 참조하지 않는다.
