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

검수 리포트의 YAML front matter에는 최소한 작업 ID, 입력 파일 식별자, 모델,
규칙 버전, RAG 출처를 기록한다. 수정 manifest에는 각 수정의 `sheet`, `cell`,
`before`, `after`, `rule_ids`, `decision`(`accept`/`partial`/`hold`)을 포함한다.
필드명·값 집합은 새로 정의하지 않고 이미 구현된
[`workbook_review_apply.py`](../agent-packages/smartthings-translation-agent/scripts/workbook_review_apply.py)의
승인 manifest 스키마를 그대로 따른다 — 자세한 정렬 방식은
[`roadmap_gemini36_colombia_rules_reports.md`](roadmap_gemini36_colombia_rules_reports.md)
§4를 참조한다. 이 문서의 "수정 manifest"는 그 도구가 받는 입력과 다른 별도 포맷이
아니라, 같은 승인 스키마로 직접 변환 가능해야 하는 상위 단계(Excel F열이 아직 없는
AI/에이전트 제안)의 산출물이다.

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

1. 사용자는 리포트 또는 manifest에서 제안 단위·시트 단위·전체를 승인하거나
   거절한다.
2. 에이전트는 승인 결과만 담은 immutable apply manifest를 생성한다. 이 manifest는
   `workbook_review_apply.py`의 `decisions` 스키마를 그대로 따르는 형태로 만든다 —
   별도 apply 스키마를 새로 설계하지 않는다.
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

## 에이전트 기능 정리와 역할

- 유지·강화: 규칙 검색, RAG DB 검색, prompt preview, Excel 구조 확인, 검수 리포트,
  승인 manifest 생성, 승인된 수정 적용과 검증.
- 축소 후보: NotebookLM처럼 긴 리포트의 보조 요약에만 쓰이고 핵심 검수·수정 경로에
  없는 기능. 제거 전 실제 사용 빈도, 대체 가능성, 유지 비용을 기록해 판단한다.
- 새 안내: `/st-setup`부터 `/st-inspect`, `/st-rag`, `/st-audit`, `/st-edit`까지
  초심자용 순서와 각 단계의 크레딧·승인 필요 여부를 하나의 workflow guide로 제공한다.

## 단계별 구현 계획

1. **공통 형식 정착**: 앱의 규칙 Markdown·검수 Markdown·수정 manifest 형식과
   파일 검증을 완성한다.
2. **읽기 전용 에이전트 검수**: 규칙/RAG 검색 및 Obsidian 리포트 작성을 지원하되
   Excel 수정 권한은 부여하지 않는다.
3. **승인 기반 수정**: manifest 승인·적용·검증을 별도 명령으로 도입하고 원본 불변을
   자동 테스트한다.
4. **병행 품질 평가**: 앱 AI, 에이전트, 사람 검수 결과를 골든셋으로 비교한다.
5. **제한적 자동화**: 정량 기준을 충족한 저위험 규칙부터 자동 반영 범위를 검토한다.

## 수용 기준

- 초심자가 workflow guide만으로 검수 준비, RAG/규칙 확인, 리포트 검토, 승인까지
  진행할 수 있다.
- 모든 제안은 셀 위치와 최소 하나의 규칙 또는 명시적 보류 사유를 가진다.
- 승인되지 않은 변경은 Excel에 적용되지 않으며, 원본 파일은 항상 보존된다.
- 적용된 변경은 manifest·리포트·수정본 Excel 사이에서 추적 가능하다.
- NotebookLM을 포함한 선택 기능의 유지 여부는 사용 근거와 대체 경로를 갖는다.
- 자동화 확대는 골든셋 및 사람 검수 비교 결과를 통과한 범위에서만 이뤄진다.
