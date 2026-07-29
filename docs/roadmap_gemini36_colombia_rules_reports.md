# Gemini 3.6 · Colombia Spanish · Rules · Markdown Reports Roadmap

## 목적

다음 네 가지 개선을 순차적으로 도입한다.

1. Gemini 3.6 Flash 지원 및 기본 모델화
2. 콜롬비아 스페인어 로케일 지원
3. 프롬프트 및 에이전트 검수 규칙의 Markdown 기반 외부화
4. 검수 리포트 Markdown 전환 및 웹 뷰어 교체

장기적으로는 앱이 생성하는 규칙·검수 결과·수정 제안을 Obsidian과
에이전트가 같은 형식으로 소비하게 하여, 번역·검수·승인된 수정 반영의
자동화를 단계적으로 지원한다. 앱은 이 전환 기간의 안정적인 실행 엔진으로
유지하며, Obsidian은 선택적 소비자이지 앱의 필수 런타임 의존성이 아니다.

이 문서는 해당 작업의 합의된 범위와 구현 기준을 기록한다. 기존
`docs/implementation_plan.md`는 이전 구현 계획으로 보존한다.

## 확정 사항

### 1. Gemini 3.6 Flash

- 모델 ID는 `gemini-3.6-flash`를 사용한다.
- 번역 모델 선택 UI의 기본값은 **Gemini 3.6 Flash (Thinking: On)** 으로 한다.
- Thinking On은 별도 budget override를 보내지 않고 모델의 기본 thinking level을 사용한다.
- API 요청, 통합 파이프라인, 텍스트 워크북, 프롬프트 미리보기, 토큰 카운팅의 기본값도 이 모델과 일치시킨다.
- Gemini 3.6에서 폐지된 sampling 파라미터(`temperature`, `top_p`, `top_k`)는 전달하지 않는다.

### 2. Colombia Spanish

- 기존 Spain Spanish는 유지한다.
- 새 canonical 언어 키: `Spanish_Colombia`
- 새 시트 코드: `CO(콜롬비아)`
- 새 glossary/locale 코드: `es_CO`
- 콜롬비아 스페인어의 기본 사용자 호칭은 `tú`로 일관되게 적용한다.
- sheet-language mapping, 프롬프트 규칙, glossary/RAG 언어 코드 경로, 데모와 UI 라벨에 이 로케일을 반영한다.

### 3. 프롬프트 및 에이전트 검수 규칙 외부화

- 변경 빈도가 높은 언어별 규칙과 BX 규칙을 런타임에서 읽는 Markdown 파일로 외부화한다.
- 파일은 YAML front matter와 사람이 읽는 Markdown 본문을 사용한다.
- front matter에는 schema version, canonical key, 표시명, 규칙 순서 및 `rules` 배열을 둔다.
- 규칙 항목은 장기적으로 `rule_id`, `severity`, `scope`, `locale`, `status`, `examples`를
  지원한다. `scope`는 최소한 `app_prompt`, `agent_audit`, `excel_apply`를 구분한다.
- 따라서 앱 프롬프트에 아직 적용하지 않는 한국어 표기, 곡선형 따옴표, disclaimer `*`
  표기 같은 회색지대 규칙도 에이전트 검수용 정책으로 명시·검색할 수 있다.
- 로더는 필수 필드, 문자열 배열, 중복 키, 지원하지 않는 schema version을 검증하고 유효하지 않은 규칙 파일은 명확히 실패 처리한다.
- 언어 alias 해석, fuzzy substring 매칭, glossary exempt/deactivation marker, bracket 정책처럼 구조와 판정이 필요한 로직은 Python 코드에 유지한다.
- 규칙의 런타임 단일 기준은 새 Markdown 규칙 파일이며, 기존 종합 규칙 문서와 agent 규칙 출처 문서도 이를 가리키도록 갱신한다.
- 적용 우선순위는 **명시 규칙·glossary > 승인된 시장별 기준 > RAG 사례**로 한다.
  번역 사례 RAG와 규칙 검색은 별도 데이터 흐름으로 유지한다.

### 4. Markdown 검수 리포트와 뷰어

- 검수 리포트는 `.txt` 대신 `.md`로 생성한다.
- 리포트는 제목, 생성 정보, 사용량 요약, 셀별 섹션, 원문/번역문/검수 결과와 JSON payload의 코드 블록을 포함한다.
- 사람용 본문과 함께 기계용 구조를 유지한다. YAML front matter에는 작업 ID, 입력 파일,
  모델, 적용 규칙 버전, RAG 출처를 기록하고, 각 셀에는 시트/셀 ID, 판정, 수정안,
  근거 규칙 ID를 기록한다.
- 수정 제안은 안정된 fenced JSON 또는 별도 JSON manifest로도 출력한다. 필드는 `sheet`,
  `cell`, `before`, `after`, `rule_ids`, `decision`(`accept`/`partial`/`hold`)을 포함한다.
  `decision`의 이름과 값 집합은 새로 정의하지 않고, 이미 구현되어 있는
  `agent-packages/smartthings-translation-agent/scripts/workbook_review_apply.py`의
  승인 manifest 스키마(`decisions[].sheet/cell/decision/final_value/reason/basis/rag_basis`)를
  그대로 따른다.
- 승인된 제안을 새 Excel 사본에 반영하는 실행 계약은 새로 만들지 않고
  `workbook_review_apply.py`를 그대로 재사용한다(원본 보존, `accept`/`partial`/`hold`만 반영,
  보호 언어·C열 diff 자동 검증까지 이미 구현됨). 다만 이 도구는 현재 `accept` 판정의 최종값을
  Excel F열에서 읽는데, 이는 원어민이 Excel에서 직접 감수한 워크플로를 전제하기 때문이다.
  F열이 없는 AI/에이전트 제안 manifest를 반영할 때는 F열 의존 없이 최종값을 함께 실어 보낼
  수 있도록 도구를 확장하거나(`accept`도 `final_value` 지정을 허용), 그 전까지는 모든 승인
  항목을 `partial` + `final_value`(=`after`)로 정규화해 적용한다. `rule_ids`는 도구의 자유
  텍스트 `basis`/`reason`/`rag_basis`를 대체하지 않고 함께 기록해 규칙 추적성을 더한다.
- 일반 작업과 텍스트 워크북 작업의 다운로드, ZIP 포함물, `/api/report/{task_id}` 응답을 Markdown 기준으로 전환한다.
- 기존 `/viewer`를 TXT delimiter parser 대신 Markdown 뷰어로 교체한다. 뷰어는
  표현 계층일 뿐 리포트 구조를 해석하는 유일한 구현체가 아니어야 한다.
- 뷰어는 `markdown-it`으로 렌더링하고 DOMPurify로 렌더링 HTML을 정화한다.
- `.md` 업로드와 `?file=/api/report/{taskId}` 자동 로드를 지원한다. 다크 모드와 PDF 내보내기는 유지한다.
- TXT 리포트 및 TXT 전용 뷰어와의 하위 호환은 유지하지 않는다.

## 구현 순서와 의존성

1. Gemini 3.6 옵션·기본값·호환성 테스트를 갱신한다.
2. Colombia Spanish 로케일을 추가한다.
3. 언어/BX 규칙 파일과 검증 로더를 도입한다. Colombia 규칙은 이 단계의 언어 규칙 파일에 포함한다.
4. 리포트 생성·다운로드·API·뷰어를 Markdown으로 전환한다.

규칙 로더는 프롬프트와 Colombia 로케일의 최종 검증에 선행해야 하며, 리포트 전환은 모델 및 로케일 변경과 독립적으로 진행할 수 있다. 수정 manifest와 승인/적용 단계는 Markdown 리포트 형식이 안정된 뒤 도입한다.

## 에이전트·Obsidian 확장 원칙

- 앱은 표준 Markdown, YAML, JSON manifest를 생성하고, Obsidian 및 에이전트는 이를
  선택적으로 소비한다. 특정 Obsidian 플러그인·CLI의 존재를 앱 동작의 전제 조건으로 두지 않는다.
- 검수는 수정 제안까지만 생성한다. 사람의 승인 이후에만 적용 manifest를 사용해 원본이 아닌
  새 Excel 사본을 수정하고, 적용 결과를 원본·규칙·검수 리포트와 연결해 검증한다.
- 향후 에이전트는 RAG 검색, 규칙 검색, Obsidian 검수 리포트 작성, 승인된 Excel 수정에
  집중한다. NotebookLM 등 사용 빈도가 낮은 기능은 이 워크플로 기여도와 운영 비용을 기준으로
  별도 정리한다.
- 앱 AI 검수 대체는 즉시 수행하지 않는다. 언어별 골든셋과 사람 검수 결과를 기준으로 앱과
  에이전트 검수를 병행하고, 검증된 단계부터 범위를 확장한다.

## 수용 기준

- UI와 서버 기본값이 `gemini-3.6-flash`이며, UI 기본 선택이 Thinking On이다.
- `Spanish`/`ES`는 Spain Spanish 규칙을 유지하고, `Spanish_Colombia`/`CO(콜롬비아)`는 `es_CO` 및 Colombian `tú` 규칙을 선택한다.
- 정상 규칙 파일은 프롬프트에 반영되고, 누락 필드·중복 키·잘못된 schema version은 테스트 가능한 오류를 낸다.
- 에이전트 검수 전용 규칙도 `scope`와 locale로 검색할 수 있으며, glossary·명시 규칙·RAG의
  우선순위가 일관되게 적용된다.
- 생성되는 검수 리포트의 확장자와 API MIME type이 Markdown이며, ZIP에도 `.md` 리포트와
  기계 판독 가능한 수정 manifest가 포함된다.
- 각 수정 제안은 시트/셀·근거 규칙·적용 전후 값을 식별할 수 있고, 승인되지 않은 제안은 Excel에 반영되지 않는다.
- `/viewer`에서 Markdown 파일 업로드와 API URL 자동 로드가 가능하며, 비신뢰 HTML은 정화된다.
