---
name: "source-command-st-help"
description: "SmartThings 번역 에이전트 최신 시작 흐름·검수 포인트·명령 안내"
---

# source-command-st-help

Use this skill when the user asks to run the migrated source command `st-help`.

## Command Template

`agent-packages/smartthings-translation-agent/SKILL.md`와 `commands/`를 근거로, 한국어로 짧고 스캔 가능하게 안내한다. 세부 명령을 기본 목록으로 나열하지 않는다.

1. **한 줄 소개**: SmartThings 다국어 번역·검수 에이전트. app 규칙·RAG·용어집·Excel workflow를 안전하게 연결하며, `Spanish_Colombia`/`CO(콜롬비아)`/`es_CO`는 기본 `tú`, `vos` 비기본, 중립 라틴아메리카 어휘를 따른다.
2. **기본 흐름**: `/st-start` → `/st-ask` 또는 `/st-review` → 사람 승인 → `/st-apply`. 일상 수정은 `/st-edit`, 비용이 드는 전체 번역/검수는 `/st-pipeline`으로 분리한다.
3. **여섯 사용자 명령**: `/st-start`(연결·다음 단계, 0), `/st-ask`(규칙·용어집·RAG, 0~), `/st-review`(읽기 전용 검수·Markdown 제안, 0), `/st-edit`(preview 후 복사본 수정, 0), `/st-apply`(승인 manifest 납품본, 0), `/st-pipeline`(승인 후 LLM 번역/audit, LLM).
4. **안전·산출물**: 원본 Excel 불변, `before` 불일치·수식·병합·보호/숨김 시트 기본 차단, draft와 glossary rich-text·전체 검증을 마친 delivery를 구분한다. RAG는 규칙·glossary보다 낮은 우선순위며, 크레딧 작업 전 승인과 secret/원본 경로 비노출을 지킨다.
5. **Excel live**: ChatGPT for Excel은 열린 workbook의 preview/draft 레이어다. glossary substring rich text는 검증 전 Delivery Python이 기준이며, live 미지원 기능은 복사본 delivery로 fallback한다고 설명한다.

관리/고급 작업(`/st-ragdb`, glossary CRUD, NotebookLM, 세부 inspect/highlight)은 필요할 때만 고급 경로로 언급한다.
