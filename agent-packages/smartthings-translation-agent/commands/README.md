# Slash Commands

이 폴더는 SmartThings translation skill용 slash command prompt 정의를 담는다.

도구별 slash command 설치 위치가 다를 수 있으므로, 이 패키지는 공통 원본만 제공한다. 설치 시 각 도구가 요구하는 commands 폴더로 필요한 파일을 복사한다.

## Primary commands

- `/st-start` → `st-start.md` — 연결 상태·다음 단계
- `/st-ask` → `st-ask.md` — 규칙·용어집·RAG 질의
- `/st-inspect` → `st-inspect.md` — 언어 시트 에이전트 검수·리포트·제안
- `/st-edit` → `st-edit.md` — 일반 Excel 수정 preview·승인·복사본 적용
- `/st-apply` → `st-apply.md` — 승인 manifest 기반 납품본 생성
- `/st-pipeline` → `st-pipeline.md` — 승인 후 LLM 번역·검수

`/st-review` → `st-review.md`는 `/st-inspect`의 전환 기간 호환 alias다. 새 안내와 자동화는
`/st-inspect <xlsx> --sheet "<언어 시트>"`를 사용한다. 기본 검수는 리드 에이전트의 2-pass이며,
5개 역할 합의 게이트는 사용자가 `--multi-agent`를 승인한 시트에서만 실행한다.

합의 게이트에서는 모든 역할이 같은 `packet_id`를 포함한 의견서를 남겨야 한다. 누락·불일치가 있거나
검수 후 셀 값이 변경되면 `changes[]`를 만들지 않고 `incomplete` 또는 `source_drift`로 사람 검토에 보낸다.

## Legacy/internal commands

아래 명령은 즉시 삭제하지 않는다. Primary command의 내부 구현 또는 고급/관리 작업으로
유지하며, workflow가 안정된 뒤 deprecated 처리한다.

- `/st-help` → `st-help.md`
- `/st-setup` → `st-setup.md`
- `/st-rules` → `st-rules.md`
- `/st-prompt` → `st-prompt.md`
- `/st-glossary` → `st-glossary.md`
- `/st-glossary-filter` → `st-glossary-filter.md`
- `/st-story-review` → `st-story-review.md`
- `/st-review-apply` → `st-review-apply.md`
- `/st-sections` → `st-sections.md`
- `/st-highlight` → `st-highlight.md`
- `/st-story-apply` → `st-story-apply.md`
- `/st-textbook` → `st-textbook.md`
- `/st-rag` → `st-rag.md`
- `/st-ragdb` → `st-ragdb.md`
- `/st-edit` → `st-edit.md`
- `/st-translate` → `st-translate.md`
- `/st-audit` → `st-audit.md`
- `/st-audit-explain` → `st-audit-explain.md`
- `/st-review-summary` → `st-review-summary.md`
- `/st-notebooklm` → `st-notebooklm.md`
- `/st-obsidian-report` → `st-obsidian-report.md`

## Notes

- command 파일은 `references/` 문서를 읽고 따르도록 설계되어 있다.
- shell command 예시는 `references/glossary-report-workflow.md`의 "셸 명령어 예시"를 본다.
