# SmartThings Translation Agent 구조 평가

> **반영 현황 (2026-08-11)**: P0(capability·anchoring), P1(실행 메타데이터), P2(escalation 기준),
> §5 배치 제어, §7 지침 계층을 구현했다. **§7은 제안된 `workflows/*/AGENTS.md` 대신 기존
> `references/` 컨벤션으로 반영했다** — `AGENTS.md`는 코딩 하네스 전용 파일명이라 `.gitignore`가
> 업로드를 막고 있고, 경로별 지침 계층 자체는 이미 존재했다. 검증 결과와 정정은 문서 끝
> "반영 결과" 참고. 미반영: §1 SKILL.md 축소(근거 부족), §6 work ledger(별도 설계 필요).

작성일: 2026-08-11  
범위: `agent-packages/smartthings-translation-agent`의 에이전트·멀티 에이전트 설계  
참고 자료: NotebookLM 노트북 **Mastering Agentic AI: Patterns and Claude Architect Strategies**

## 목적

현재 SmartThings Translation Agent가 루트 에이전트에 과도한 도구와 책임을 노출하고 있는지,
그리고 `/st-inspect`의 리드-서브에이전트 검수 구조가 멀티 에이전트 설계 원칙에 부합하는지를
1차 진단한다.

NotebookLM 자료는 외부 참고 분석이다. 이 문서의 현재 구조 판단은 패키지의 `SKILL.md`,
`commands/`, `scripts/agent_*` 구현을 직접 확인한 사실을 기반으로 한다.

## 핵심 결론

1. **도구 노출면은 과밀하다.** 기능 자체가 과도하다기보다, 서로 다른 위험도와 업무 단위의 기능이
   루트 에이전트의 단일 지시 표면에 함께 노출되어 있다.
2. **멀티 에이전트 검수 구조의 기본 방향은 좋다.** 역할 분리, 공통 근거 패킷, 읽기 전용 의견서,
   보수적 병합, 사람 검토 큐가 이미 갖춰져 있다.
3. **다음 개선의 우선순위는 역할 추가가 아니다.** 역할별 실제 권한 최소화, 동조(anchoring) 경고의
   승격 차단 연결, 실행 완결성 메타데이터가 우선이다.

## 1. 루트 에이전트의 도구 노출 평가

### 확인한 현황

| 항목 | 수량 | 평가 |
| --- | ---: | --- |
| 사용자 slash command | 27개 | 루트 도구 선택지로는 많음 |
| `scripts/` Python 파일 | 98개 | 내부 구현·작업 이력용이 상당수 포함됨 |
| `SKILL.md`에서 직접 언급한 실행 스크립트 | 22개 | 에이전트의 경로 선택 부담이 큼 |
| `SKILL.md` | 205줄 | 기능, 안전 규칙, 레거시 경로가 혼재 |

`commands/README.md`는 이미 다음을 primary entry point로 정의한다.

- `/st-start`
- `/st-ask`
- `/st-inspect`
- `/st-edit`
- `/st-apply`
- `/st-pipeline`

그러나 `SKILL.md`는 레거시·고급 명령까지 상세히 나열한다. 그 결과 아래 경로들이 함께 노출된다.

- 검수: `/st-inspect`, `/st-review`, `/st-story-review`
- 적용: `/st-apply`, `/st-review-apply`, `/st-story-apply`, `/st-highlight`
- 비용형 파이프라인: `/st-pipeline`, `/st-translate`, `/st-audit`
- 지식 질의: `/st-ask` 아래 rules, glossary, RAG, prompt 세부 경로

### NotebookLM 원칙과 대조

참고 자료는 단일 에이전트에 모든 도구를 넣는 방식을 안티패턴으로 보고, 한 역할에는 한 가지 업무와
필요한 1~2개 도구만 제공하는 전문화를 권고한다. 너무 넓은 도구·컨텍스트는 도구 선택 정확도를
낮추고 토큰 비용을 증가시킨다는 취지다.

현재 패키지는 다음 구조로 정리하는 것이 적절하다.

```text
루트 라우터: st-start / st-ask / st-inspect / st-edit / st-apply / st-pipeline
    └─ 내부 전문 워크플로: legacy·관리·세부 command 및 script
        └─ 작업 이력·일회성 script: agent-facing surface에서 제외
```

즉, 기존 기능을 제거할 필요는 없다. 루트 에이전트가 primary command만 선택하고, 내부 라우터가
검증된 세부 스크립트를 선택하도록 해야 한다.

## 2. 멀티 에이전트 구조 평가

### 현재 구조

`/st-inspect`는 기본적으로 리드 에이전트의 2-pass 검수로 실행된다. 5개 역할의 병렬 검수는
사용자가 `--multi-agent`로 승인한 escalation 경로다.

| 구성 요소 | 현재 동작 | 평가 |
| --- | --- | --- |
| 오케스트레이션 | 리드가 근거 패킷 생성·역할 배정·결과 병합 | 적합 |
| 전문 역할 | 문법/유창성, 의미 충실도, 현지화/톤, 스타일·하드룰 예외, story/UI | 적합 |
| 입력 격리 | 역할별 프롬프트에는 공통 근거 패킷만 포함, 타 역할 의견 미포함 | 적합 |
| 출력 계약 | 역할별 JSON opinion, `packet_id`, `stance`, `after`, 제약 상태 | 적합 |
| 병합 | 2개 이상 지지 + 반대 없음일 때만 주관적 제안 후보 생성 | 적합 |
| 불완전 처리 | 역할 누락·패킷 불일치 시 모든 제안을 사람 검토 큐로 보류 | 강점 |
| 하드룰 | 결정론적 constraint가 `blocked`/`human_review`면 승격 차단 | 강점 |

이 구조는 NotebookLM 자료의 핵심 원칙과 부합한다.

- 서브에이전트는 한 관점만 담당한다.
- 다른 서브에이전트의 추론 과정이나 의견을 전달하지 않는다.
- 공통 근거를 구조화된 데이터 계약으로 전달한다.
- 원시 작업 과정을 리드 컨텍스트로 누적하지 않고 의견서 결과만 병합한다.
- 불일치·불완전·제약 위반은 사람 검토로 이관한다.

## 3. 멀티 에이전트 개선 우선순위

### P0 — 역할별 실제 capability profile 적용

현재 프롬프트는 다른 의견을 보지 말라고 지시하지만, 실행 환경에서 역할별 접근 가능한 도구와 파일이
정말로 제한되는지는 별도 강제가 필요하다.

- 서브에이전트: 읽기 전용 packet, 허용된 RAG 조회, JSON 의견서 기록만 허용
- 리드: dispatch, packet 검증, merge만 수행
- 적용/납품/Excel 쓰기/외부 MCP: 서브에이전트 capability에서 제외
- 각 역할에는 필요한 근거 슬라이스만 전달하여 패킷 크기도 축소

### P0 — anchoring을 승격 차단에 연결

현재 `detect_role_anchoring()`은 한 역할의 support 집합이 다른 역할을 반복하는 상황을 경고하지만,
경고 자체는 제안 승격을 막지 않는다.

동조가 감지된 제안은 다음 중 하나를 적용한다.

1. `human_review_queue`로 이동한다.
2. 동조하지 않은 제3 역할의 독립 지지를 요구한다.
3. 특정 역할 조합의 합의 가중치를 낮춘다.

두 역할이 정상적으로 같은 결론을 낼 수 있으므로, 처음에는 자동 거절보다 **제3 역할 재검증**이
안전한 기본값이다.

### P1 — 실행 완결성 및 신뢰도 계약 추가

NotebookLM 자료는 불완전 종료와 낮은 신뢰도의 결과를 사람에게 넘길 것을 권고한다. 현재 계약은
`completed`/`no_findings`, 역할 누락, packet mismatch는 처리하지만 다음을 기록하지 않는다.

- `run_id`, agent/model/version
- 시작·종료 시각, 오류, timeout, stop reason
- 입력 packet hash와 실제 사용한 RAG evidence ID
- 역할별 confidence 또는 uncertainty reason
- token/context budget 초과 여부

이 항목이 없거나 비정상 종료면 역할 누락과 동일하게 `incomplete` 처리해야 한다.

### P2 — 멀티 에이전트 escalation 기준을 명문화

사용자 승인으로만 멀티 에이전트를 켜는 현재 방식은 비용·지연 통제에는 좋다. 다만 리드가 언제
사용자에게 escalation을 제안해야 하는지 기준을 추가해야 한다.

- UI 조건·면책 문구처럼 검토 축이 교차하는 경우
- 리드 2-pass에서 의미/톤 판단이 충돌하는 경우
- 고위험 locale 또는 반복 오류 패턴이 있는 경우
- 자동 적용 전 사람 검토 큐가 임계치를 넘는 경우

## 4. 권장 목표 구조

```text
사용자
  → 루트 라우터 (6개 primary command)
    → 리드 에이전트 (작업 계획·승인·packet 생성·병합)
      → [필요 시] 역할별 격리 서브에이전트
          - 최소 capability
          - 역할별 데이터 슬라이스
          - JSON 결과 + 실행 메타데이터
      → 결정론적 검증·anchoring 판정
      → 승인 대기 제안 또는 human_review_queue
  → 사람 승인
  → 적용 전용 워크플로
```

## 5. 배치 처리 적용 평가

### NotebookLM 참고 분석

참고 자료는 실시간 응답이 필요하지 않은 대량 프롬프트를 비동기 Batch API로 제출하면 비용을
낮출 수 있다고 설명한다. 또한 배치 작업은 독립된 단위로 분할하고, 하위 작업의 원시 컨텍스트는
상위 세션으로 가져오지 않으며, `stop_reason`과 부분 실패를 추적해 실패 단위만 재처리해야 한다고
권고한다.

자료에서 특정 제공사의 Batch API 비용·지연 조건을 언급한 부분은 해당 제공사의 당시 서비스
맥락에 한정된 참고 정보다. 이 프로젝트의 Gemini/GPT 호출에 같은 조건을 그대로 적용해서는 안
된다. 제공사별 Batch API 지원 여부, 비용, 최대 지연은 도입 결정 전에 공식 문서로 별도 확인한다.

### 현재 구현 확인

| 대상 | 현재 상태 | 평가 |
| --- | --- | --- |
| 파일 단위 번역 배치 | `batch_co_rollout.py`가 파일 간 순차 처리, 파일별 원자적 manifest 기록, 결과 셀 수 검증 수행 | 안전한 기반이 있음 |
| 앱 검수 배치 | `es_co_app_inspect_batch.py`가 워크북별 순차 검수와 report manifest 기록 수행 | 안전한 기반이 있음 |
| 대량 RAG 수집 | `story053_semantic_rag_batch.py`가 완료한 `(sheet, row)`를 식별하고 checkpoint부터 재개 | 재개 가능성이 구현됨 |
| 워크북 내부 LLM 병렬성 | `workbook_translate.py`/`workbook_audit.py`가 `max_concurrency`를 `TranslationChecker`에 전달 | 제한 동시성은 구현됨 |
| 제공사 Batch API | app `src/`에서 제출·polling·결과 매핑 구현을 찾지 못함 | 현재는 미도입 |

`batch_co_rollout.py`는 작업 결과를 매 파일 뒤 manifest에 원자적으로 기록하지만, 새 실행 시 기존
manifest를 읽어 성공 단위를 건너뛰지는 않는다. 따라서 완료 이력은 남지만 파일 배치 전체에 대한
resume/멱등성은 아직 보장하지 않는다.

### 적용 판단

**적용한다.** 다만 첫 단계는 제공사 Batch API가 아니라, 현재 파일 단위 배치의 제어 계층을
강화하는 것이다.

- 배치 단위를 `workbook + target sheet + workflow version`으로 고정한다.
- 각 단위에 입력 해시, glossary/model/version, 비용 추정, 상태를 기록한다.
- 상태는 최소 `pending`, `running`, `succeeded`, `failed`, `needs_review`로 구분한다.
- 재실행 시 `succeeded`는 건너뛰고 실패·중단 단위만 명시적으로 재시도한다.
- 배치 중에는 읽기 전용 분석·번역 후보 생성까지만 수행한다.
- Excel 적용은 사람 승인 manifest를 받은 뒤 별도의 순차 작업으로 실행한다.

멀티 에이전트 검수도 같은 원칙을 따른다. `시트 × 역할`을 무제한 병렬화하지 않고, packet 생성과
역할별 읽기 전용 검토에만 제한된 동시성을 적용한다. 결과 병합, 사람이 검토할 항목의 확정,
Excel 적용은 순차적으로 유지한다.

### 제공사 Batch API 도입 전제

다음 조건을 충족할 때만 긴급하지 않은 대량 번역·검수에 비동기 제공사 Batch API를 검토한다.

1. 제출 후 지연된 결과를 받는 비동기 job 계약을 정의한다.
2. 요청 단위와 결과를 workbook/sheet/cell 및 input hash로 안정적으로 매핑한다.
3. partial failure, timeout, 취소, 재제출을 manifest 상태로 추적한다.
4. 기존 `max_concurrency` 실시간 경로와 Batch 경로의 비용·지연·품질을 파일럿으로 비교한다.
5. 사람 승인과 Excel 적용을 Batch 결과 수신 이후의 별도 단계로 유지한다.

## 6. 세션 간 연속성: Obsidian, manifest, handoff

### 현재 구조 평가

현재 구조는 특정 번역 검수의 연속성에는 강점이 있다.

- Obsidian 리포트는 finding, 규칙, 판단, approval 상태, Decision Log처럼 사람이 읽는 맥락을 남긴다.
- approval/delivery manifest는 변경, 승인, 검증, 납품의 기계적 사실을 남긴다.
- `obsidian_workflow.py sync-status`는 유효한 result manifest가 있을 때만 Obsidian 리포트를
  `applied` 상태로 갱신한다.
- Obsidian Base는 report frontmatter를 조회해 `draft`, `reviewed`, `approved`, `applied` 상태를
  사람이 탐색하게 한다.

따라서 아래 흐름에는 충분한 기반이 있다.

```text
특정 workbook 검수
→ report / approval manifest 생성
→ 다음 세션에서 승인 판단 확인
→ Excel 적용
→ delivery manifest 검증
```

그러나 장기·다중 세션 작업의 단일 재개 기준으로는 부족하다. 리포트와 manifest가 존재하더라도
상위 목표, 현재 단계, 다음 행동, 차단 요인, 관련 산출물 전체를 한 번에 가리키는 work-level
index와 handoff 계약은 강제되지 않는다.

### 권장 역할 분리

| 계층 | 역할 | 기준 |
| --- | --- | --- |
| Manifest | 변경·실행·검증의 기계적 사실 | 작업별 불변 증적 |
| Obsidian report | 판단 근거와 사람이 읽는 맥락 | 검색·의사결정·회고 |
| Work ledger / handoff | 현재 단계와 다음 행동 | 세션 재개용 단일 진입점 |

### 최소 보강안

각 작업에 안정적인 `work_id`를 부여하고, 다음 구조를 사용한다.

```text
outputs/work/<work_id>/
  state.json                 # 현재 상태의 기계적 기준
  handoff.md                 # 다음 세션이 읽는 1페이지 요약
  evidence_packet.json
  review_report.md
  approval_manifest.json
  delivery_manifest.json
```

`state.json`에는 최소 `work_id`, objective, status, source workbook/sheet/input hash,
관련 artifact 경로, 핵심 결정, blocker, `next_action`을 기록한다.

Obsidian 리포트 frontmatter에는 `work_id`, `status`, `report_id`, `source_workbook`, `packet_id`,
approval/delivery manifest 경로, `next_action`을 추가해 작업 노트와 산출물을 링크한다.

세션 시작 시에는 전체 대화 원문이 아니라 `handoff.md`, `state.json`, 필요한 manifest만 읽는다.
세션 종료 시에는 state의 status, blockers, next_action을 갱신한다. 이 방식으로 Obsidian은
기억·탐색 계층, manifest는 증적 계층, work ledger/handoff는 세션 재개 계층을 맡는다.

## 7. 경로별 계층형 지침과 폴더 구조

### NotebookLM에서 확인한 원칙

참고 자료는 Claude Code의 `CLAUDE.md`를 예로 들어 지침을 한 파일에 집중하지 않고 다음
3단계로 계층화할 것을 설명한다.

1. 프로젝트 최상위 레벨
2. 프로젝트/패키지 폴더 레벨
3. 하위 디렉터리 레벨

직접 확인된 원칙은 작업 경로에 가까워질수록 더 구체적인 지침을 적용해 시스템의 동작을 제어하는
것이다. 자료는 또한 서브태스크를 독립된 영역으로 격리하고, 각 에이전트에 필요한 데이터 슬라이스만
전달해야 한다고 설명한다.

아래의 구체적인 폴더명과 역할 분리는 이 원칙을 SmartThings Translation Agent에 적용한 설계
제안이며, 자료가 직접 지정한 구조는 아니다.

### 권장 구조

```text
SmartThings_translation_web_app/
├── AGENTS.md                         # 전역 보안·승인·원본 불변·공통 개발 규칙
├── agent-packages/
│   └── smartthings-translation-agent/
│       ├── AGENTS.md                 # 패키지 경계·app 연결·공통 실행 규약
│       ├── SKILL.md                  # 루트 라우터: 주 업무 흐름만 안내
│       ├── workflows/
│       │   ├── knowledge/
│       │   │   └── AGENTS.md         # 규칙·glossary·RAG·NotebookLM: 읽기 전용
│       │   ├── review/
│       │   │   └── AGENTS.md         # packet·리드 검수·서브에이전트·병합 규칙
│       │   ├── edit/
│       │   │   └── AGENTS.md         # preview·승인·draft 사본 규칙
│       │   ├── delivery/
│       │   │   └── AGENTS.md         # 승인 manifest·검증·납품 규칙
│       │   └── operations/
│       │       └── AGENTS.md         # RAG DB·glossary 관리·배치·비용 규칙
│       ├── scripts/                  # 워커가 호출하는 구현
│       └── outputs/work/<work_id>/   # state·handoff·manifest·report
```

### 계층별 책임

| 계층 | 포함할 내용 | 포함하지 말아야 할 내용 |
| --- | --- | --- |
| app root `AGENTS.md` | 시크릿, 승인, 원본 보존, 테스트·코딩 공통 규칙 | 번역 workflow의 세부 도구 설명 |
| package `AGENTS.md` | app 연결, portability, 공통 artifact/manifest 계약 | 개별 작업의 역할 프롬프트 |
| root `SKILL.md` | 사용자 의도 라우팅, primary entry point, 안전 게이트 | 모든 legacy command와 script 사용법 |
| `workflows/*/AGENTS.md` | 워커의 목적, 허용 capability, 입출력 계약, escalation | 다른 워커의 내부 도구·원시 컨텍스트 |
| `outputs/work/<work_id>` | 작업별 상태, handoff, 근거, report, manifest | 다른 작업의 임시 파일·공유 scratch 파일 |

이 구조는 root `SKILL.md`를 짧게 유지하고, 필요한 workflow의 지침과 도구만 작업 시점에 읽게 한다.
따라서 기능을 삭제하지 않고도 루트 에이전트의 선택 부담과 컨텍스트 과밀을 줄일 수 있다.

## 반영 결과 (2026-08-11 구현)

### 검증된 사실과 정정

수치는 모두 확인됐다(slash command 27, `scripts/*.py` 98, `SKILL.md` 205줄).

**§7의 전제는 두 가지 점에서 틀렸다.**

1. `AGENTS.md`는 이미 루트(58줄)와 패키지(8줄)에 존재한다. 루트 것은 번역 규칙이 아니라 멀티
   에이전트 오케스트레이션 랩 설명이며 `CLAUDE.md`와 거의 중복이다.
2. **`AGENTS.md`/`CLAUDE.md`/`gemini.md`는 `.gitignore`에서 "에이전트 지침 문서(로컬 유지,
   업로드 금지)"로 지정돼 있다.** 이들은 이 저장소를 유지보수하는 *코딩 하네스*의 설정 파일이지
   배포되는 스킬 패키지의 콘텐츠가 아니다. 따라서 제품 지침에 이 파일명을 쓰면 안 된다.

그리고 §7이 요구한 "경로별 계층형 지침"은 **이미 충족돼 있다**. 패키지는 `references/*.md`로
워크플로별 지침을 분리해 두었고(9개, `SKILL.md`가 27회 참조), 이미 `*-workflow.md` 네이밍
컨벤션을 쓴다. `excel-workflow.md`는 제안된 `edit/`, `self-vs-pipeline.md`는 `operations/`의
내용을 이미 담고 있다.

실제로 빠져 있던 것은 **다중 에이전트 검수 워크플로 문서 하나**뿐이었다. `workflows/` 트리를
새로 만드는 대신 기존 컨벤션에 맞춰 `references/review-workflow.md`를 추가하고, 배치 resume
규약은 `references/self-vs-pipeline.md`에 덧붙였다. `SKILL.md`에는 작업→지침 라우팅 표를 넣었다.

### P0 — anchoring 승격 차단 (반영)

경고에 그치던 것을 **승격 차단**으로 연결했다. 제안한 세 선택지 중 **제3 역할 재검증**을
채택했다: 동조 역할의 지지는 독립으로 세지 않고, 남은 독립 관점이 2개 미만이면
`anchored_support_needs_independent_role`로 사람 검토 큐에 보낸다. 자동 거절이 아니다.

구현 중 탐지기의 두 결함을 발견해 함께 고쳤다.

1. **부분 에코 미탐지** — 전부 베낀 경우(부분집합)만 잡아, 하나만 빼고 베끼면 통과했다.
   중복률 임계값(기본 0.8)을 추가했다.
2. **다중 동조 누락** — 한 역할이 여러 역할을 동조해도 첫 번째만 기록하고 멈췄다. 그 결과
   ES_CO에서 차단되어야 할 3건 중 1건만 걸렸다. 모든 쌍을 기록하도록 고쳤다.

ES_CO 회귀 검증: 제안 21 → **18**, 큐 43 → 46. 차단된 3건은 사전 수동 계산과 정확히 일치한다.
이 과정에서 `style_and_hard_rule_exceptions`가 `localization_tone`과 **11/11 완전 일치**한다는
사실이 새로 드러났다(기존 인지: `story_and_ui_coherence` 8/8 하나뿐).

### P0 — capability profile (부분 반영, 한계 명시)

- **반영**: 역할별 근거 슬라이싱(`ROLE_EVIDENCE_SLICES`). resolver 내부 카드와 candidate overlay는
  이를 소유한 역할에만 전달하고, 나머지 역할은 하드룰이 걸렸다는 사실만 요약으로 받는다.
  프롬프트에 허용/금지 행동을 명시적으로 선언한다.
- **한계**: 실제 도구 권한 제한은 코드가 아니라 에이전트 런타임(하네스) 설정 영역이다. 프롬프트
  선언만으로는 강제되지 않으며, 이는 이번에 앵커링이 발생한 것과 같은 실패 방식이다. 런타임
  권한 격리가 가능한지는 별도 확인이 필요하다.

### P1 — 실행 완결성 메타데이터 (반영)

의견서에 `run_id`/`agent`/`model`/시각/`stop_reason`/`confidence`/`rag_evidence_ids`/`error`를
받고 `agent_runs`에 기록한다. **환경이 제공한 값만 기록한다.** 오류가 있거나 `stop_reason`이
정상 종료가 아니면 역할 누락과 동일하게 `incomplete` 처리한다.

### P2 — escalation 기준 (반영)

`commands/st-inspect.md`와 `workflows/review/AGENTS.md`에 명문화했다. 리드는 escalation을
**제안만** 하고 스스로 켜지 않는다(`--multi-agent`는 여전히 사용자 승인 전용).

### §5 — 배치 제어 (반영)

`batch_co_rollout.py --resume`을 추가했다. 이전 manifest에서 `ok`/`prepped`만 건너뛰고
`error`/`skipped`는 다시 시도한다 — 일시적 실패를 영구 실패로 굳히지 않기 위해서다.
제공사 Batch API는 도입하지 않았고, 전제 조건을 `workflows/operations/AGENTS.md`에 남겼다.

### §7 — 지침 계층 (기존 구조로 반영)

새 트리를 만들지 않고 기존 `references/` 컨벤션을 따랐다(위 "검증된 사실과 정정" 참고).

- `references/review-workflow.md` 신설 — 다중 에이전트 검수의 유일하게 없던 지침
- `references/self-vs-pipeline.md`에 배치 resume 규약 추가
- `SKILL.md` 상단에 작업 → 지침 라우팅 표 추가

### 미반영과 사유

- **§1 SKILL.md 축소**: 라우팅 표만 추가하고 본문은 줄이지 않았다. 205줄을 덜어내면 안전 규칙이
  누락될 위험이 있고, "도구가 많아 선택이 부정확하다"는 가설이 이 저장소에서 아직 측정되지
  않았다. 워크플로 계층이 자리잡은 뒤 실제 오선택 사례를 근거로 진행하는 편이 안전하다.
- **§6 work ledger / handoff**: 새 산출물 표면(`outputs/work/<work_id>/`)을 만드는 작업이라
  기존 report/manifest 계약과의 관계를 먼저 설계해야 한다. 별도 작업으로 남긴다.

## 근거 파일

- `agent-packages/smartthings-translation-agent/SKILL.md`
- `agent-packages/smartthings-translation-agent/commands/README.md`
- `agent-packages/smartthings-translation-agent/commands/st-inspect.md`
- `agent-packages/smartthings-translation-agent/scripts/agent_sheet_review.py`
- `agent-packages/smartthings-translation-agent/scripts/agent_role_prompts.py`
- `agent-packages/smartthings-translation-agent/scripts/agent_review_contract.py`
- `agent-packages/smartthings-translation-agent/scripts/agent_sheet_merge.py`
- `agent-packages/smartthings-translation-agent/scripts/batch_co_rollout.py`
- `agent-packages/smartthings-translation-agent/scripts/es_co_app_inspect_batch.py`
- `agent-packages/smartthings-translation-agent/scripts/story053_semantic_rag_batch.py`
- `agent-packages/smartthings-translation-agent/scripts/workbook_translate.py`
- `agent-packages/smartthings-translation-agent/scripts/workbook_audit.py`
- `agent-packages/smartthings-translation-agent/scripts/obsidian_workflow.py`
- `agent-packages/smartthings-translation-agent/commands/st-obsidian-report.md`
