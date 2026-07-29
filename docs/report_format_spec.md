# 번역 검수 리포트 공통 양식 명세

## 목적

앱과 에이전트가 같은 Markdown 리포트를 생성·읽을 수 있도록 공통 형식을 정의한다.
이 양식은 기존 TXT 리포트의 검수 정보를 보존하면서, 각 셀의 **원문**, **현재 번역문**,
**제안 번역문**을 명확히 구분한다.

앱은 정형화된 기본 리포트를 생성한다. 에이전트는 이를 기반으로 분석·조사·결정 기록을
자유롭게 추가하거나 수정할 수 있지만, Excel 자동 적용에 사용되는 구조화 필드는 안정적으로
유지한다.

관련 계획:

- [앱·규칙·Markdown 리포트 로드맵](roadmap_gemini36_colombia_rules_reports.md)
- [Obsidian 기반 에이전트 운영 로드맵](roadmap_obsidian_agent_workflow.md)

## 핵심 원칙

- **현재 번역문**은 보고서 생성 시점에 Excel 또는 앱 파이프라인에 실제로 존재하는 값이다.
- **제안 번역문**은 검수 결과가 권장하는 변경값이다. 수정이 필요 없으면 빈 문자열 대신
  `제안 없음`을 기록한다.
- 리포트 생성과 검수는 Excel을 수정하지 않는다.
- Excel 반영은 `apply_status: approved`인 항목만, 원본이 아닌 새 Excel 사본에 적용한다.
- 모든 변경 제안은 셀 위치, 근거 규칙, 적용 전후 값을 식별할 수 있어야 한다.
- 명시 규칙·glossary·시장 기준·RAG 사례의 우선순위는 별도 규칙 명세를 따른다.

## 문서 구조

```text
YAML front matter       # 기계 판독용 작업 메타데이터
# 리포트 제목
## 요약                 # 사람용 전체 요약
## 셀 검수              # 안정된 셀별 구조
### {시트} · {셀}
  YAML finding block    # 상태, 근거 규칙, 적용 상태
  #### 원문
  #### 현재 번역문
  #### 제안 번역문
  #### 변경 이유
  #### 검수 상세
  #### 원본 검수 Payload
## Agent Notes          # 선택적, 자유 서술 영역
## Decision Log         # 선택적, 승인·보류 이력
```

## 리포트 메타데이터

모든 리포트는 YAML front matter로 시작한다.

```yaml
---
report_schema_version: 1
report_id: review-20260729-001
workflow: app_review # app_review | agent_review | combined_review
status: draft # draft | reviewed | approved | applied
source_file_id: story-039
generated_at: 2026-07-29T14:30:00+09:00
translation_model: gemini-3.6-flash
audit_model: gpt-5.4-mini
rules_version: 2026-07-29
rag_sources: []
---
```

`source_file_id`는 업로드 파일명 또는 비밀 경로 대신 안전한 작업 식별자를 사용한다.

## 셀 검수 양식

각 셀은 고유 anchor와 기계 판독 가능한 YAML 블록을 가진다.

````md
### DE(독일) · C15 {#de-c15}

```yaml
finding_id: DE-C15
revision: 1
status: needs_revision # pass | warning | needs_revision | blocked
apply_status: pending_approval # not_applicable | pending_approval | approved | rejected | applied
rule_ids:
  - de-register-du
  - glossary-capitalization
rag_evidence_ids:
  - rag-de-0021
```

#### 원문

```text
Make home care easier with SmartThings.
```

#### 현재 번역문

```text
Mache die Pflege deines Zuhauses mit SmartThings einfacher.
```

#### 제안 번역문

```text
Mit SmartThings wird die Pflege deines Zuhauses einfacher.
```

#### 변경 이유

- 문장 흐름을 자연스럽게 조정했습니다.
- `SmartThings` 표기는 용어집 기준을 유지했습니다.

#### 검수 상세

- 대소문자: 별도 지적 사항 없음
- 용어집: 준수
- RAG 일관성: 유사 DE 사례 2건 참고
- AI 검수: Needs Revision

#### 원본 검수 Payload

```json
{
  "grade": "Needs Revision",
  "suggested_fix": "Mit SmartThings wird die Pflege deines Zuhauses einfacher."
}
```
````

렌더러는 각 `yaml`, `text`, `json` code block을 독립적으로 처리한다.

## 기존 TXT 및 뷰어 필드 대응

| 기존 TXT/HTML 필드 | Markdown 위치 | 비고 |
| --- | --- | --- |
| `[상세 - 원문]` / `sourceText` | `#### 원문` | 변경 없이 보존 |
| `[상세 - 번역문]` / `targetText` | `#### 현재 번역문` | 실제 Excel/파이프라인 값 |
| 없음 | `#### 제안 번역문` | AI 또는 에이전트의 수정 권고 |
| `[상세 - 대소문자 점검]` / `casingCheck` | `#### 검수 상세` | 항목별 bullet |
| `[상세 - 용어집 점검]` / `glossaryCheck` | `#### 검수 상세` | 항목별 bullet |
| `[상세 - RAG 일관성 참고]` / `ragCheck` | `#### 검수 상세` 및 `rag_evidence_ids` | 근거 ID 연결 |
| `[상세 - 역번역]` | `#### 검수 상세` | 필요할 때 text block 추가 |
| `[상세 - AI 검수 결과]` / `geminiQa` | `#### 검수 상세` | 사람용 결과 |
| `[상세 - AI Payload]` | `#### 원본 검수 Payload` | 원본 JSON 보존 |

## 앱과 에이전트의 역할

### 앱 리포트

- 필수 메타데이터와 모든 셀의 정형 섹션을 생성한다.
- 현재 번역문, 검사 결과, 원본 AI payload를 기록한다.
- AI 수정안이 있으면 제안 번역문으로 기록하고 `apply_status`는 항상
  `pending_approval`로 시작한다.
- 제안이 없으면 제안 번역문에 `제안 없음`, `apply_status`에 `not_applicable`을 기록한다.

### 에이전트 리포트

- 앱 리포트를 읽고 같은 `report_id`, `finding_id`를 유지한 채 분석을 보강할 수 있다.
- `## Agent Notes`, `## Market Research`, `## Decision Log`, 사용자 요청별 섹션은 자유롭게
  추가·편집할 수 있다.
- 제안 번역문, `apply_status`, `rule_ids`를 변경하면 `revision`을 증가시키고 Decision Log에
  변경 이유와 시간을 남긴다.
- 구조화 셀 블록을 삭제하거나 셀 위치·적용 전후 값을 자유 서술만으로 대체해서는 안 된다.

## 승인 및 수정 manifest

Excel 적용 전에는 리포트의 승인 결과를 별도 manifest로 고정한다.

```json
{
  "manifest_schema_version": 1,
  "report_id": "review-20260729-001",
  "source_file_id": "story-039",
  "changes": [
    {
      "finding_id": "DE-C15",
      "sheet": "DE(독일)",
      "cell": "C15",
      "before": "Mache die Pflege deines Zuhauses mit SmartThings einfacher.",
      "after": "Mit SmartThings wird die Pflege deines Zuhauses einfacher.",
      "rule_ids": ["de-register-du"],
      "approval_status": "approved"
    }
  ]
}
```

적용 도구는 `approval_status: approved` 항목만 처리하고, 적용 전 `before` 값이 실제 Excel과
일치하는지 확인한다. 불일치 시 수정하지 않고 검증 실패로 보고한다.

### 적용 계약 (확정)

**`approval_status`와 `decision`은 서로 다른 축이며 하나로 합치지 않는다.**

| 필드 | 소속 | 의미 | 값 |
| --- | --- | --- | --- |
| `approval_status` | 이 문서의 승인 manifest | **승인 게이트** — 사람이 결재했는가 | `pending_approval` / `approved` / `rejected` / `applied` / `not_applicable` |
| `decision` | `workbook_review_apply.py`의 실행 manifest | **어느 텍스트가 최종인가** | `accept` / `partial` / `hold` |

한 행이 `approved`이면서 동시에 `partial`일 수 있다. 이름을 통일하면 두 정보 중 하나가
사라지므로 각자 유지한다.

**변환은 도구가 직접 한다.** 별도 어댑터 스크립트를 두지 않는다 — 실행을 잊을 수 있는
중간 단계를 만들지 않기 위해서다. `workbook_review_apply.py`는 manifest에 `decisions`
배열이 있으면 실행 manifest로, `changes` 배열이 있으면 이 문서의 승인 manifest로 인식해
자동 변환한다(`decisions_from_report_changes()`).

변환 규칙:

| 승인 manifest | → 실행 manifest |
| --- | --- |
| `approval_status: approved` | `decision: accept` |
| 그 외 모든 상태 | **변환 대상에서 제외** (Excel에 절대 반영되지 않음) |
| `after` | `final_value` |
| `before` | `expected_before` |
| `rule_ids` | `rule_ids` + `basis`(표시용 문자열) |
| `finding_id` | `finding_id` (결과 manifest까지 추적) |

승인 항목에 `before` 또는 `after`가 없으면 변환이 실패한다. 승인된 항목이 하나도 없으면
도구는 아무것도 쓰지 않고 오류를 낸다.

**F열 의존 제거.** `accept`는 이제 두 경로를 모두 지원한다.

- `final_value`가 있으면 그 값을 쓴다 → AI/에이전트 제안 경로. F열이 아예 없어도 된다.
- `final_value`가 없으면 F열을 읽는다 → 기존 원어민 Excel 감수 경로(동작 변경 없음).

**드리프트 가드.** `expected_before`가 있으면 적용 직전 실제 C열 값과 대조하고, 다르면
셀을 쓰지 않고 중단한다. 검수 시점과 적용 시점 사이에 워크북이 바뀐 경우를 잡는다.
이 검증은 원래 이 문서가 요구했지만 도구에 구현되어 있지 않았고, 이번에 추가됐다.
결과 manifest의 각 항목에 `before_verified`로 검증 수행 여부가 기록된다.

## 뷰어 요구사항

- Markdown heading과 코드 블록을 안전하게 렌더링한다.
- `finding_id`, `status`, `apply_status`를 읽어 기존 HTML 카드와 같은 요약·필터 UI를 제공할 수 있다.
- `현재 번역문`과 `제안 번역문`을 나란히 보여 변경 여부를 즉시 확인할 수 있게 한다.
- YAML/JSON 구조는 뷰어가 숨기거나 접을 수 있지만, 파일 내용에서는 보존한다.
- 비신뢰 HTML은 DOMPurify로 정화한다.
