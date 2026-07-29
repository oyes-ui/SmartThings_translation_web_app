# Obsidian 기반 번역 검수 에이전트 운영 로드맵

## 목표

SmartThings 번역 업무를 초심자도 안전하게 수행할 수 있도록, 규칙·RAG·검수
결과·Excel 수정 지시를 Markdown 중심으로 연결한다. 최종 목표는 번역, 검수,
승인된 수정 반영의 자동화 가능성을 검증하는 것이며, 앱 AI 검수를 즉시 제거하지
않고 사람 검수와 병행해 신뢰도를 쌓는다.

앱의 데이터 계약과 기반 구조는
[`roadmap_gemini36_colombia_rules_reports.md`](roadmap_gemini36_colombia_rules_reports.md)를
따른다. 이 문서는 그 계약을 사용하는 에이전트 워크플로와 운영 원칙을 정의한다.

## 운영 경계

- 앱은 표준 Markdown, YAML front matter, JSON manifest를 생성하는 실행 엔진으로
  유지한다. Obsidian 설치, 플러그인, CLI는 앱 실행의 필수 조건이 아니다.
- 에이전트는 원본 Excel을 직접 수정하지 않는다. 먼저 검수 리포트와 수정 제안
  manifest를 생성하고, 사람의 승인을 받은 manifest만 새 Excel 사본에 적용한다.
- 우선순위는 **명시 규칙·glossary > 승인된 시장별 기준 > RAG 사례**다. RAG는
  근거와 일관성 참고 자료이며 규칙을 덮어쓰지 않는다.
- API 키·개인정보·원본 파일 경로는 노트, report, manifest, agent 로그에 기록하지
  않는다.

## 공통 산출물 계약

| 산출물 | 생성 주체 | 핵심 내용 | 다음 단계 |
| --- | --- | --- | --- |
| 규칙 Markdown | 정책 담당자 / 에이전트 | `rule_id`, locale, scope, severity, 예시 | 규칙 검색·프롬프트·검수 |
| RAG 검색 결과 | 앱 또는 에이전트 | 일치 사례, 출처, locale, 검색 점수 | 검수 근거 |
| 검수 리포트 Markdown | 앱 또는 에이전트 | 작업 메타데이터, 셀별 판정, 근거 규칙, 수정안 | 사람 검토·Obsidian 기록 |
| 수정 manifest JSON | 에이전트 | 시트·셀, 적용 전/후 값, 근거 규칙, 승인 상태 | 승인된 Excel 수정 |
| 수정본 Excel | 적용 도구 | 승인된 manifest만 반영한 새 파일 | 검증·전달 |

검수 리포트와 수정 manifest의 정확한 필드는 새로 정의하지 않고
[`report_format_spec.md`](report_format_spec.md)를 단일 기준으로 따른다(YAML front
matter, 셀별 finding 블록, `changes[].sheet/cell/before/after/rule_ids/approval_status`
승인 manifest). 이 manifest는 Excel F열이 아직 없는 AI/에이전트 제안 단계의 산출물이며,
이미 구현된
[`workbook_review_apply.py`](../agent-packages/smartthings-translation-agent/scripts/workbook_review_apply.py)의
`decision`(`accept`/`partial`/`hold`) 승인 스키마와 이름·값 집합이 다르다. 두 스키마는
아직 자동 변환되지 않으며, 이 간극을 어떻게 메울지는
[`roadmap_gemini36_colombia_rules_reports.md`](roadmap_gemini36_colombia_rules_reports.md)
§4 "미해결 간극"에서 결정한다.

## 사용자 워크플로

### Level 1 — 검수 준비 및 탐색 (크레딧 0)

1. Excel 파일·대상 시트·용어집을 확인한다.
2. 대상 locale의 명시 규칙과 관련 RAG 사례를 검색한다.
3. 에이전트가 적용 규칙과 검수 범위를 Markdown으로 요약한다.

### Level 2 — 검수와 리포트 작성

1. 앱 또는 에이전트가 셀별 번역·검수 결과를 생성한다.
2. 에이전트는 결과를 Obsidian 호환 Markdown 리포트로 정리한다.
3. 각 수정 제안에 원문, 현재 번역, 제안 번역, 근거 규칙 ID, RAG 근거,
   신뢰도와 보류 사유를 남긴다.
4. 규칙과 RAG가 충돌하거나 근거가 부족한 항목은 자동 수정 후보가 아니라
   사람 결정 항목으로 표시한다.

### Level 3 — 승인과 Excel 반영

일상적인 단일/복수 셀 수정은 `/st-edit`의 빠른 경로로 처리한다. 이 경로도 현재값과
제안값을 preview로 먼저 보여주고, 사용자 승인 뒤 원본이 아닌 새 Excel 사본만 만든다.
적용 직전 `before` 값 검증과 적용 후 workbook diff 검증을 필수로 하며, 수식 셀·병합 셀·
보호/숨김 시트는 기본 차단한다. C열 변경 후 재하이라이트를 생략한 결과는 `draft`이며
납품본이 아니다.

감수·검수 제안을 최종 납품본에 반영할 때는 `/st-apply`의 승인 manifest 경로를 사용한다.

1. 사용자는 리포트 또는 manifest에서 제안 단위·시트 단위·전체를 승인하거나
   거절한다.
2. 에이전트는 승인 결과만 담은 immutable apply manifest를 생성한다. Excel에 실제로
   반영할 때는 `report_format_spec.md`의 승인 manifest를
   `workbook_review_apply.py`가 받는 `decisions` 스키마로 변환한다(§4 "미해결 간극"
   결정에 따름) — 별도 apply 도구를 새로 만들지 않는다.
3. 적용 도구(`workbook_review_apply.py`)는 원본을 보존하고 timestamp가 붙은 새 Excel
   파일에만 수정한다.
4. 적용 후 시트/셀 값, rich text, 공백, 서식 보존 여부와 manifest 일치 여부를
   검증하고 결과를 리포트에 추가한다.

### Level 4 — 제한적 자동화 평가

- 언어별 골든셋과 사람 검수 결과를 기준으로 에이전트 제안의 정확도·규칙 준수율·
  잘못된 수정 비율을 측정한다.
- 기준을 충족한 낮은 위험 범위(예: 명시 glossary 불일치, 확정된 대소문자 규칙)부터
  자동 승인 후보로 확대한다.
- 의미·톤·시장 문화 판단이 필요한 항목은 사람 승인 경로를 유지한다.

## Obsidian 및 스킬 활용

`kepano/obsidian-skills`의 표준 Agent Skills를 채택 후보로 사용한다.

- `obsidian-markdown`: rules, 검수 리포트, 승인 기록을 Obsidian Flavored Markdown으로
  작성·수정한다.
- `obsidian-cli`: vault 내 파일 탐색, 링크, 검색, 상태 확인을 수행한다.
- `obsidian-bases`: 작업·리포트·승인 상태의 목록, 필터, 요약을 제공한다.
- `defuddle`: 외부 규정·공식 언어 자료를 Markdown으로 정리할 때만 선택적으로 사용한다.

해당 스킬은 Markdown, Bases, JSON Canvas, Obsidian CLI를 다루며 Codex를 포함한
skills-compatible agent에서 사용할 수 있도록 제공된다.
[kepano/obsidian-skills 공식 저장소](https://github.com/kepano/obsidian-skills)

## 사용자 명령 구조

초심자에게는 다음 여섯 명령만 노출한다.

| 명령 | 목적 | 내부로 흡수되는 기존 기능 |
| --- | --- | --- |
| `/st-start` | 연결 상태와 다음 단계 안내 | help, setup |
| `/st-ask` | 규칙·용어집·RAG 사례 질의 | rules, glossary, glossary filter, rag, prompt, audit explain |
| `/st-review` | 읽기 전용 검수와 리포트/제안 manifest 생성 | inspect, sections, story review, Obsidian report, review summary |
| `/st-edit` | 일반 Excel 셀 수정의 preview·승인·복사본 적용 | edit |
| `/st-apply` | 승인 manifest 기반의 감수본·납품본 생성 | review apply, story apply, highlight |
| `/st-pipeline` | 승인 후 LLM 번역 또는 검수 | translate, audit |

기존 세부 명령은 구현 도구로 먼저 흡수하고, 새 진입점과 workflow guide가 안정된 뒤
deprecated 처리한다. 용어집 CRUD, RAG DB 빌드, 텍스트 workbook 생성은 관리자/고급 작업으로
숨긴다. NotebookLM은 핵심 경로가 아닌 선택 기능으로 유지 여부를 사용 빈도와 대체 가능성으로
결정한다.

## 에이전트 기능 정리와 역할

- 유지·강화: 규칙 검색, RAG DB 검색, prompt preview, Excel 구조 확인, 검수 리포트,
  일반 Excel 수정의 preview/승인/diff 검증, 승인 manifest 생성, 승인된 수정 적용과 검증.
- 축소 후보: NotebookLM처럼 긴 리포트의 보조 요약에만 쓰이고 핵심 검수·수정 경로에
  없는 기능. 제거 전 실제 사용 빈도, 대체 가능성, 유지 비용을 기록해 판단한다.
- 새 안내: `/st-start` → `/st-ask` → `/st-review` → 사람 승인 → `/st-apply`의
  초심자용 순서와 각 단계의 크레딧·승인 필요 여부를 하나의 workflow guide로 제공한다.
  `/st-edit`는 이 흐름과 별도로 일상적인 Excel 수정에 사용하는 빠른 경로이며,
  `/st-pipeline`은 명시적 사용자 승인 후에만 사용한다.

## 단계별 구현 계획

1. **공통 형식 정착**: 앱의 규칙 Markdown·검수 Markdown·수정 manifest 형식과
   파일 검증을 완성한다.
2. **읽기 전용 에이전트 검수**: 규칙/RAG 검색 및 Obsidian 리포트 작성을 지원하되
   Excel 수정 권한은 부여하지 않는다.
3. **승인 기반 수정**: manifest 승인·적용·검증을 별도 명령으로 도입하고 원본 불변을
   자동 테스트한다.
4. **병행 품질 평가**: 앱 AI, 에이전트, 사람 검수 결과를 골든셋으로 비교한다.
5. **제한적 자동화**: 정량 기준을 충족한 저위험 규칙부터 자동 반영 범위를 검토한다.
6. **명령 통합과 편집 안정화**: 여섯 사용자 명령으로 안내를 단순화하고, `/st-edit`의
   dry-run·before 검증·원본 불변·rich text 재하이라이트·workbook diff 검증을 자동화한다.

## 수용 기준

- 초심자가 workflow guide만으로 검수 준비, RAG/규칙 확인, 리포트 검토, 승인까지
  진행할 수 있다.
- 모든 제안은 셀 위치와 최소 하나의 규칙 또는 명시적 보류 사유를 가진다.
- 승인되지 않은 변경은 Excel에 적용되지 않으며, 원본 파일은 항상 보존된다.
- 적용된 변경은 manifest·리포트·수정본 Excel 사이에서 추적 가능하다.
- 일반 `/st-edit` 수정도 승인 전 preview와 적용 후 workbook diff를 남기며, `draft`와
  재하이라이트/검증을 마친 납품본을 구분한다.
- NotebookLM을 포함한 선택 기능의 유지 여부는 사용 근거와 대체 경로를 갖는다.
- 자동화 확대는 골든셋 및 사람 검수 비교 결과를 통과한 범위에서만 이뤄진다.
