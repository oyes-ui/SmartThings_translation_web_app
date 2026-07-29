---
description: SmartThings 번역 에이전트 개요·검수 포인트·명령 안내
---

`/st-help`는 `/st-start`와 같은 시작 안내를 제공하는 호환 명령이다. 한국어로 짧고
스캔 가능하게 아래를 안내한다.

1. SmartThings 다국어 번역·검수를 돕는 app-aware 에이전트이며, app의 규칙·RAG·glossary·
   Excel 파이프라인을 호출하고 재구현하지 않는다고 설명한다.
2. 기본은 셀프 모드(크레딧 0), 대량 작업은 사용자 승인 뒤 파이프라인 모드(LLM)라고 구분한다.
3. 검수는 문법/자연스러움, 의미 충실도, glossary, 현지화, 대소문자, 포맷·BX의 6항목과
   story 맥락을 본다고 설명한다.
4. 원본 Excel 불변, 수정 전 preview·승인, API 비용 사전 확인, secret 미노출을 강조한다.

사용자에게는 다음 여섯 명령을 먼저 제시한다.

| 명령 | 용도 | 크레딧 |
| --- | --- | --- |
| `/st-start` | 연결 상태와 다음 단계 | 0 |
| `/st-ask` | 규칙·glossary·RAG 질의 | 0~ |
| `/st-review` | 읽기 전용 검수·리포트·수정 제안 | 0 |
| `/st-edit` | 일반 Excel 수정 preview·승인·복사본 적용 | 0 |
| `/st-apply` | 승인 manifest 기반 납품본 | 0 |
| `/st-pipeline` | 승인 후 LLM 번역·검수 | LLM |

세부 명령은 고급/관리 또는 위 명령의 내부 구현으로 남아 있으며 즉시 삭제하지 않았다고
안내한다.
