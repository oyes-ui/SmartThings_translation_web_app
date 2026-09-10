---
description: 시작하기·현재 작업 확인
argument-hint: [상태 확인 또는 이어서 할 작업]
---

기본 명령 목록과 한 줄 설명은 `commands/README.md`를 따른다.
처음 시작할 때 `scripts/bootstrap.py --json`으로 연결을 읽기 전용 확인하고 다음 행동을 안내한다.
진행 중인 작업은 `references/command-execution.md`의 상태/재개 규칙을 따른다.
현재 작업이 있으면 처음부터 다시 시작하거나 동일한 작업을 새로 만들지 않는다.
“어디까지 됐어?”는 상태·blocker·다음 행동만 읽고, “이어서 해줘”는 입력·계약·기존 승인을
검증한 뒤 해당 작업의 동일 실행 명령을 사용한다. 새 승인을 추론하지 않는다.

업무 안내의 정본인 `references/workflow-guide.md`와 `commands/README.md`를 읽고, 사용자의 입력·원하는 산출물에 맞는 시작점과 다음 행동만 짧게 안내한다. 완료한 단계는 생략하며 판단 가능한 사항은 다시 질문하지 않는다.
