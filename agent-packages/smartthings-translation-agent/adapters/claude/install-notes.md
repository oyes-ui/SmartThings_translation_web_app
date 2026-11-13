# Claude Code 설치 노트

공통 본체는 [SKILL.md](../../SKILL.md), 사용자 명령은 [명령 안내](../../commands/README.md)가 정본이다.
기본 명령은 `st-start`, `st-ask`, `st-translate`, `st-inspect`, `st-edit`, `st-apply`의 여섯 개다.
고급·호환 명령의 전체 개수나 환경별 설치 상태를 고정해서 안내하지 않는다.

## 이 저장소의 명령 연결

앱 저장소의 `.claude/commands/`는 패키지의 공통 `commands/*.md`를 읽도록 하는 진입 문서다.
저장소 루트에서 사용하며 실행·승인 절차는 공통 문서에 둔다. 예전 설치본은 공통 문서로
위임하는지 확인한다. `commands/README.md`는 실행 명령이 아닌 색인이다.

## 본체를 별도로 설치하는 경우

Project-local 예시는 `.claude/skills/smartthings-translation-agent/`, 사용자 전역 예시는
`~/.claude/skills/smartthings-translation-agent/`다. 설치 대상이 공통 본체를 찾는지 확인하고
`references/`, `scripts/`, `commands/`를 함께 유지한다. 같은 절차를 여러 위치에서 따로 관리하지 않는다.

설치 후에는 패키지 폴더를 실행 루트로 사용하고 앱 저장소를 명시적으로 연결한다.

```bash
cd <installed-skill-folder>
python scripts/bootstrap.py --app-root /path/to/SmartThings_translation_web_app --json
python scripts/bootstrap.py --app-root /path/to/SmartThings_translation_web_app --save
```

의존성·데이터 전달은 [이식성 안내](../../references/portability.md)를 따른다. daemon은 필요 없다.

## 동작 확인

`/st-start` 또는 “SmartThings 번역 에이전트 시작해줘”로 연결 상태를 확인한다. 읽기 전용 예시:

```bash
python scripts/workbook_inspect.py <story.xlsx> --sheet "JA(일본)" --sections --json
```

검수는 [st-inspect](../../commands/st-inspect.md), 승인된 draft·납품·재개는
[실행 규약](../../references/command-execution.md)을 따른다. 준비 스크립트만 실행했다고 검수가 완료되지는 않는다.
