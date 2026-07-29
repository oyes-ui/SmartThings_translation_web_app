# Gemini 3.6 · Colombia Spanish · Rules · Markdown Reports Roadmap

## 목적

다음 여섯 가지 개선을 도입한다. 아래 번호는 항목 식별용(§1~§6)이며 적용 순서가
아니다. 실제 순서는 "구현 순서와 의존성"을 따른다.

1. Gemini 3.6 Flash 지원 및 기본 모델화
2. 콜롬비아 스페인어 로케일 지원
3. 프롬프트 및 에이전트 검수 규칙의 Markdown 기반 외부화
4. 검수 리포트 Markdown 전환 및 웹 뷰어 교체
5. 에이전트 명령 통합과 안전한 일반 Excel 수정
6. ChatGPT for Excel 전용 레이어

이 중 **§2(콜롬비아 스페인어 로케일 지원)가 이번 업데이트의 핵심 목표**다. 나머지
다섯 항목은 같은 시기에 함께 해결하려는 부수 개선이다.

단, **§3(규칙 Markdown 외부화)은 §2의 선행 조건이다.** 콜롬비아 규칙을 Python 상수로
먼저 쓴 뒤 §3에서 다시 md로 옮기는 이중 작업을 피하기 위해, §3 위에 md 규칙 파일로
한 번에 작성한다. §3가 지연되면 §2도 함께 지연됨을 감수한다. §1·§4·§5·§6은 §2를
막지 않는다.

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

- 기존 Spain Spanish(`Spanish`/`ES`)는 규칙과 코드를 유지한다.
- 새 canonical 키 `Spanish_Colombia`, 시트 코드 `CO(콜롬비아)`, glossary/locale 코드 `es_CO`.
- sheet-language mapping, 프롬프트 규칙, glossary/RAG 경로, 데모/UI 라벨에 반영한다.
- 호칭은 `tú`를 SmartThings CO 브랜드 보이스의 의도적 기본값으로 삼는다("콜롬비아
  표준"이 아님). 기존 Spain Spanish 규칙과 같은 패턴(텍스트 규칙, 별도 입력 필드
  없음)을 따른다.
- 실제 프롬프트에 들어가는 규칙 원문은 Spain/German 수준으로 짧게 유지한다. 판단이
  필요한 항목(활용 일관성, vos 탐지 등)은 프롬프트에 넣지 않고 `agent_audit`
  scope 체크리스트로 분리한다(§3).

**`app_prompt` 규칙 원문** (§3 형식의 `rules/languages/Spanish_Colombia.md`에 작성.
`prompt_modules.py`에는 콜롬비아 항목을 추가하지 않는다):
1. tú 기본, 소스/프로젝트가 명시할 때만 usted.
2. vos/voseo는 브리프가 명시하지 않는 한 금지.
3. 중립/라틴아메리카 어휘(`celular`, `computador`) 사용, Spain 어휘(`móvil`, `ordenador`,
   `vosotros`)·강한 지역색 구어체(`ratico`, `momentico`) 회피.
4. 복수 상대는 `vosotros` 대신 `ustedes` + 3인칭 복수 활용.

**`agent_audit` 전용 체크리스트** (§3 `scope: agent_audit`, 프롬프트에는 안 넣음):
- tú 활용(동사·소유사 `tu`·목적격 `te`·명령형) 일관성 확인. 단 주어 생략된 짧은
  명사형 UI 라벨(예: "Configuración")은 대상에서 제외.
- vos/voseo 발견 시 브리프의 명시적 허용 여부 확인.

**RAG 격리**: `rag_retriever.py`가 이미 `target_lang` 정확 일치로 조회하므로
(`WHERE ... target_lang = ?`), CO가 고유 `target_lang` 값을 쓰면 코드 변경 없이
ES와 자동 격리된다. `es_CO` 사례를 우선하고 없으면 일반 Spanish를 낮은 우선순위로
참고한다.

**범위**: 번역/검수 프롬프트와 glossary/RAG 식별자에 한정한다. 날짜·통화(COP)·주소
같은 i18n은 이 앱이 렌더링하지 않는 영역이라 제외한다.

### 3. 프롬프트 및 에이전트 검수 규칙 외부화 — ✅ 구현 완료

구현 결과 확정된 사항(계획 대비 추가):

- 규칙 항목은 문자열이 아니라 **객체**다: `rule_id`(필수, 전역 유일), `text`(필수),
  `scope`(필수, 리스트). `severity`/`status`/`locale`/`examples`는 선택.
  마이그레이션된 69개 규칙에는 아무도 분류하지 않은 `severity`를 임의로 넣지 않았다.
- `rule_order: sequence` — 배열 순서가 곧 모델이 보는 순서라는 계약 선언.
  v1 로더는 이 값만 허용한다.
- `scope`는 리스트다. 한 규칙이 `[app_prompt, agent_audit]` 둘 다일 수 있다.
  `agent_audit` 전용 규칙은 앱의 번역·검수 프롬프트 **양쪽 모두에서 제외**되며,
  `rules_loader.get_rules().rules_for_scope("agent_audit")`로만 조회된다.
- 본문(body)은 앱이 파싱하지 않는다. 규칙 문구를 본문에 중복 기재하지 않는다
  (규칙 수정이 이 기능의 최빈 작업이므로 편집 지점을 하나로 유지).
- `{canonical_key}.md`이므로 4개 파일명에 공백이 들어간다(`Simplified Chinese.md` 등).
  언더스코어 정규화는 `English_US`/`French_Canada`의 의미를 깨뜨리므로 쓰지 않는다.
- 레거시 상수는 삭제하지 않고 **마이그레이션 등가성 테스트의 기준값**으로 활용했다
  (`tests/test_rules_migration_equivalence.py`). 이 테스트는 레거시 상수 삭제 커밋에서
  함께 삭제한다.



- 변경 빈도가 높은 언어별 규칙과 BX 규칙을 런타임에서 읽는 Markdown 파일로 외부화한다.
- 파일은 YAML front matter와 사람이 읽는 Markdown 본문을 사용한다.
- **파일 위치(앱·에이전트 공용 계약)**: `src/translation_web_app/rules/languages/{canonical_key}.md`
  (언어별), `src/translation_web_app/rules/bx_style.md`(BX)에 둔다. 앱 패키지 경계
  안에 둬야 앱만 단독 배포(Hugging Face 등)할 때도 함께 배포된다. 에이전트는 새
  접근 경로를 만들지 않고 기존 `--app-root` 관례로 같은 경로를 읽는다(예:
  `workbook_review_apply.py --app-root`와 동일 패턴). 이 위치는 앱과 에이전트 양쪽에
  영향을 주므로 변경 시 이 로드맵과 `rules-sources.md`를 함께 갱신한다.
- front matter에는 schema version, canonical key, 표시명, 규칙 순서 및 `rules` 배열을 둔다.
- 규칙 항목은 장기적으로 `rule_id`, `severity`, `scope`, `locale`, `status`, `examples`를
  지원한다. `scope`는 최소한 `app_prompt`, `agent_audit`, `excel_apply`를 구분한다.
- 따라서 앱 프롬프트에 아직 적용하지 않는 한국어 표기, 곡선형 따옴표, disclaimer `*`
  표기 같은 회색지대 규칙도 에이전트 검수용 정책으로 명시·검색할 수 있다.
- 로더는 필수 필드, 문자열 배열, 중복 키, 지원하지 않는 schema version을 검증하고 유효하지 않은 규칙 파일은 명확히 실패 처리한다.
- **로딩 시점**: 프로세스 시작 시 1회 로드해 메모리에 캐시한다. 검증 실패 시 앱
  기동을 중단한다(요청 처리 중 깨진 규칙이 조용히 쓰이지 않도록). 요청마다
  재읽기는 하지 않는다 — 상시 구동되는 FastAPI 서버에서 매 프롬프트 생성마다
  파일 I/O·YAML 파싱을 반복할 이유가 없고, `scripts/prompt_preview.py` 같은 CLI
  도구는 매번 새 프로세스로 실행되므로 어차피 최신 파일을 읽는다. 규칙 수정 후
  반영에는 앱 재시작이 필요하며, 기존 수작업 이중 문서 동기화보다 가벼운 비용이라
  v1에서는 hot-reload를 요구하지 않는다.
- 언어 alias 해석, fuzzy substring 매칭, glossary exempt/deactivation marker, bracket 정책처럼 구조와 판정이 필요한 로직은 Python 코드에 유지한다.
- 규칙의 런타임 단일 기준은 새 Markdown 규칙 파일이며, 기존 종합 규칙 문서와 agent 규칙 출처 문서도 이를 가리키도록 갱신한다.
- **기존 Python 상수는 삭제하지 않고 레거시로 유지한다**: 마이그레이션된 언어의
  `LANGUAGE_LOCALIZATION_RULES`/`BX_STYLE_RULES` 등 기존 항목을 `prompt_modules.py`에서
  지우지 않는다. 다만 런타임에서는 참조하지 않는다 — 활성 소스는 항상 새 md 파일
  하나여야 두 시스템이 동시에 값을 제공해 다시 드리프트가 생기는 걸 막는다. 레거시
  상수는 md 로더가 프로덕션에서 안정성이 확인된 뒤 별도 정리 커밋으로 제거한다(§5
  여섯 명령 통합의 "즉시 삭제 안 함" 원칙과 동일).
- 적용 우선순위는 **명시 규칙·glossary > 승인된 시장별 기준 > RAG 사례**로 한다.
  번역 사례 RAG와 규칙 검색은 별도 데이터 흐름으로 유지한다.

### 4. Markdown 검수 리포트와 뷰어

- 검수 리포트는 `.txt` 대신 `.md`로 생성한다. 정확한 문서 구조, YAML front matter,
  셀별 finding 블록, 앱/에이전트 역할 구분은 [`report_format_spec.md`](report_format_spec.md)를
  단일 기준으로 따른다 — 이 로드맵에는 필드를 다시 정의하지 않는다.
- 일반 작업과 텍스트 워크북 작업의 다운로드, ZIP 포함물, `/api/report/{task_id}` 응답을
  Markdown 기준으로 전환한다.
- 기존 `/viewer`를 TXT delimiter parser 대신 Markdown 뷰어로 교체한다. `markdown-it`으로
  렌더링하고 DOMPurify로 정화하며, `.md` 업로드와 `?file=/api/report/{taskId}` 자동 로드를
  지원한다. 다크 모드와 PDF 내보내기는 유지한다. TXT 리포트 및 TXT 전용 뷰어와의 하위
  호환은 유지하지 않는다.
- **✅ 해소됨 (계약 확정)**: 아래 간극은 `report_format_spec.md`의 "적용 계약 (확정)"
  절로 확정되고 `workbook_review_apply.py`에 구현·테스트되었다. 결론은 (a)/(b) 중
  하나가 아니라 둘 다였다 — 두 필드는 의미가 다르므로 이름을 통일하지 않고 각자 유지하되
  (`approval_status`=승인 게이트, `decision`=최종 텍스트 선택), 도구가 두 형식을 모두
  직접 읽어 변환하고(어댑터 스크립트 없음), `accept`가 `final_value`를 받아 F열 의존을
  없앴다. 추가로, 이 문서가 요구했지만 도구에 없던 `before` 드리프트 검증
  (`expected_before`)을 구현했다. §4와 apply 확장을 별도 브랜치로 나눠도 되는 시점은
  지금부터다.

  <details><summary>원래 기록된 간극 (이력용)</summary>

- **미해결 간극**: `report_format_spec.md`의 승인 manifest는 `approval_status`
  (`approved`/…)로 AI 제안의 승인 여부를 표시하고, 이미 구현된
  `agent-packages/smartthings-translation-agent/scripts/workbook_review_apply.py`는
  `decision`(`accept`/`partial`/`hold`) 3단 판정을 쓰며 `accept`의 최종값을 Excel F열에서
  읽는다(원어민이 Excel에서 직접 감수하는 워크플로 전제). 두 스키마는 이름과 값 집합이
  다르고 지금은 서로 자동 변환되지 않는다. Excel 적용을 실제로 구현할 때 (a)
  `report_format_spec.md` manifest → `workbook_review_apply.py` 입력 변환 어댑터를 만들거나,
  (b) `workbook_review_apply.py`가 `final_value`를 가진 `accept`도 받도록 확장해 F열 의존을
  없앤다. 이 결정 전까지 두 문서를 "이미 정합됨"으로 취급하지 않는다.

  </details>

### 5. 에이전트 명령 통합과 안전한 일반 Excel 수정

- 사용자에게 노출하는 에이전트 명령은 다음 여섯 개로 통합한다.
  - `/st-start`: app 연결 상태, 사용 가능한 기능, 다음 단계 안내
  - `/st-ask`: 규칙, 용어집, RAG 사례 질의
  - `/st-review`: 읽기 전용 통합 검수와 Markdown 리포트/수정 제안 manifest 생성
  - `/st-edit`: 일반적인 단일 또는 복수 Excel 셀 수정
  - `/st-apply`: 승인 manifest 기반의 감수본/납품본 생성
  - `/st-pipeline`: 사용자 승인 후 앱 LLM 번역 또는 검수 실행
- 기존 세부 slash command는 즉시 삭제하지 않는다. 위 여섯 진입점의 내부 구현 도구로
  흡수한 뒤 사용 경로가 안정되면 deprecated 처리한다. 용어집 CRUD, RAG DB 빌드,
  텍스트 workbook 생성은 관리자/고급 작업으로 분리한다.
- `/st-review`는 읽기 전용이며 `report_format_spec.md` 계약의 리포트와 제안 manifest까지만
  생성한다. `/st-apply`는 승인된 제안을 최종 납품본에 반영하는 경로로 한정한다.
- `/st-edit`는 검수 승인과 무관한 일상 수정의 빠른 경로다. 기본 동작은 dry-run이며,
  현재값·제안값·영향 범위·rich text/용어집 하이라이트 위험을 먼저 보여준다. 사용자 승인
  전에는 Excel 파일을 만들거나 수정하지 않는다.
- 실제 편집은 원본을 절대 덮어쓰지 않고 새 파일에만 적용한다. edit manifest에는 최소
  `sheet`, `cell`, `before`, `after`를 기록하고, 적용 직전 실제 값이 `before`와 다르면
  중단한다. 수식 셀, 병합 셀, 보호/숨김 시트는 기본 차단하고 별도 경고와 승인을 요구한다.
- 적용 후에는 허용된 셀 외 변경 없음, 수식·병합·보호 시트 보존, 파일 열기 가능 여부를
  workbook diff로 검증한다. C열 번역문 변경 후 필요한 glossary 재하이라이트를 생략한
  산출물은 `draft`로 표시하며 납품본으로 안내하지 않는다.
- `delivery` 편집은 납품 범위 전체와 필요한 KR/US source sheet의 재하이라이트 및 텍스트
  보존 검증을 강제한다. 단순 `/st-edit`와 승인 기반 `/st-apply`는 모두 이 원본 불변·승인·
  diff 검증 원칙을 공유한다.

### 6. ChatGPT for Excel 전용 레이어

- 기존 `smartthings-translation-agent`의 공통 규칙·RAG 우선순위·승인 정책은 유지하고,
  ChatGPT for Excel에서 실행할 전용 레이어를 별도 패키지로 둔다. 제안 구조는 다음과 같다.

  ```text
  smartthings-translation-agent/
  ├─ SKILL.md                     # 공통 워크플로·규칙·안전 원칙
  ├─ references/                  # locale·RAG·manifest 공통 기준
  ├─ excel-chatgpt/
  │  ├─ SKILL.md                  # 열린 workbook 대상 진입 규칙
  │  ├─ edit-workflow.md          # preview → 승인 → 적용 → 검증
  │  ├─ officejs-capabilities.md  # 지원 API·requirement set·fallback
  │  └─ manifest-schema.md        # edit/review/apply 공통 계약 참조
  └─ scripts/                     # Codex/CLI·backend 전용 구현
  ```

- Excel 전용 레이어는 로컬 Python 실행, 로컬 파일 경로, `openpyxl`을 전제로 하지 않는다.
  현재 열린 workbook·활성 sheet·선택 range를 읽고, 변경안을 보여준 뒤 승인된 범위만
  Office.js/ChatGPT for Excel의 live workbook 기능으로 수정한다.
- RAG, glossary, 규칙 검색은 SmartThings app의 API/MCP가 제공한다. Excel 전용 skill은
  그 결과를 해석하고 편집 workflow에 연결하지만 DB 또는 secret을 직접 다루지 않는다.
- edit manifest와 review/apply manifest의 의미는 `report_format_spec.md`를 공통 기준으로
  유지한다. live Excel 편집 후에도 `before` 재확인, 허용 범위 diff, 수식 오류 확인,
  사용자에게 보여 줄 변경 요약을 수행한다.
- Office.js 기능은 requirement set을 런타임에서 확인한다. 요구 API가 없거나 workbook의
  rich text 보존 여부를 확신할 수 없으면 쓰기를 중단하고 기존 CLI/복사본 경로로
  fallback한다. 이벤트는 세션 재시작 뒤 재등록해야 하므로 영구 감사 로그의 유일한 근거로
  사용하지 않는다.
- 첫 PoC는 `CO(콜롬비아)` 시트의 C열 2~3개 셀로 제한한다. glossary rich text,
  병합/보호 시트, 수식 재계산, before/after diff를 Codex/CLI 파일 경로와 live Excel
  경로에서 각각 검증한다. PoC를 통과하기 전에는 live Excel 결과를 자동 납품본으로
  취급하지 않는다.

#### Glossary 하이라이팅 투트랙

두 경로는 같은 glossary 버전과 edit manifest를 사용하며, 결과의 신뢰 수준만 다르다.

| 경로 | 목적 | glossary 공급 방식 | 산출물과 게이트 |
| --- | --- | --- | --- |
| **Live Excel** | 현재 열린 workbook에서 매칭 preview·일상 수정·결과 확인 | 사용자가 명시적으로 가져온 CSV를 숨김/보호 `__ST_GLOSSARY` table에 저장하거나, 승인된 app API/MCP에서 조회 | 매칭 목록, `before`/`after`, 적용 범위 diff. Office.js가 rich-text run 보존을 지원·검증한 경우에만 부분 문자열 하이라이트를 적용한다. 그렇지 않으면 수정 없이 후보만 보여 주고 delivery 경로로 보낸다. |
| **Delivery Python** | 납품본의 정확한 용어 단위 rich text 하이라이트와 파일 단위 무결성 검증 | 기존 glossary CSV를 `openpyxl` 하이라이터가 읽음 | 원본 불변 복사본, rich text 하이라이트, 전체 납품 범위와 KR/US source sheet 검증, highlight report. 이 경로가 rich text의 기준 구현이다. |

- Live Excel에서 임의의 로컬 경로를 자동 탐색하거나 `.env`·DB를 읽지 않는다. CSV는 사용자가
  명시적으로 선택/가져오기 해야 하며, 저장 시 원본 경로 대신 glossary 버전·checksum·locale만
  manifest에 기록한다.
- `/st-edit --highlight-glossary`는 새 사용자 명령을 추가하지 않고 `/st-edit`의 하위
  workflow로 제공한다. `preview`가 기본이며, 실제 반영 전 매칭 용어·대상 셀·사용 glossary
  버전을 보여준다.
- Live Excel의 Office.js capability 또는 rich text 보존 검증이 실패하면 실패를 숨기거나
  셀 전체 색상으로 대체하지 않는다. 사용자가 명시적으로 셀 단위 강조를 선택하지 않은 한
  Delivery Python 경로를 제안한다.

## 구현 순서와 의존성

1. 언어/BX 규칙 파일과 검증 로더(§3)를 먼저 완성한다. 기존 로케일(Spain, German 등)의
   Python 상수를 이 형식으로 이관한다.
2. Colombia Spanish(§2)를 완성된 로더 위에 md 규칙 파일로 직접 작성한다. Python
   상수로 먼저 쓰고 나중에 옮기는 이중 작업은 하지 않는다.
3. Gemini 3.6 옵션·기본값·호환성 테스트(§1)를 갱신한다. §2/§3과 건드리는 파일이
   겹치지 않아 독립적으로 병렬 진행할 수 있다.
4. 리포트 생성·다운로드·API·뷰어(§4)를 Markdown으로 전환한다. 모델·로케일 변경과
   무관하게 별도로 진행할 수 있으나 `checker_service.py`를 §1과 함께 건드리므로
   병합 순서를 조율한다(아래 "병렬 작업 리스크" 참고).
5. `report_format_spec.md`와 `workbook_review_apply.py`의 manifest 변환/`final_value`
   계약을 확정한다. 이후 `/st-review`와 `/st-apply`가 같은 계약으로 제안·승인·적용한다.
6. 여섯 사용자 명령으로 진입점을 통합하고 `/st-edit`의 dry-run, before 검증,
   재하이라이트, workbook diff 검증을 구현한다.
7. ChatGPT for Excel 전용 레이어(§6) PoC를 수행한다. 공통 규칙/manifest를 재사용하고,
   Office.js capability fallback과 rich text 보존 검증을 통과한 범위만 live edit로 확대한다.
8. glossary 하이라이팅 투트랙을 구현한다. `__ST_GLOSSARY` import/버전 관리와 Live Excel
   preview를 먼저 만들고, Delivery Python 산출물과 동일한 입력에서 매칭·텍스트 보존 결과를
   비교한다.

위 순서 번호는 실행 순서이며 §번호와 다르다. §2가 최우선 목표라는 원칙은 유지하되,
그 실현 경로는 "§3을 우회해 빠르게"가 아니라 "§3 위에 한 번에 제대로 짓기"다(목적
섹션과 동일한 결정).

## 병렬 작업 리스크

여러 에이전트가 브랜치를 나눠 동시에 진행할 때의 리스크.

- **안전하게 병렬 가능**: §1(Gemini 3.6)과 §3(규칙 로더)는 건드리는 파일이 겹치지
  않는다(§1은 `checker_service.py`/`model_handler.py`/`main.py`의 기본값, §3은
  `prompt_modules.py`/`prompt_builder.py`). 두 브랜치를 동시에 진행해도 충돌 위험이
  낮다.
- **순차 필수**: §2(Colombia)는 §3 완료를 전제한다. §3이 머지되기 전에 §2 브랜치를
  시작하면 로더 스키마가 바뀔 때마다 다시 작업해야 한다.
- **같은 파일, 조율 필요**: §1과 §4는 둘 다 `checker_service.py`/`main.py`를 건드린다
  (§1은 기본 모델 파라미터, §4는 리포트 조립·다운로드 엔드포인트). 영역은 다르지만
  동시에 열어두면 머지 순서를 정하고 나중 브랜치가 리베이스하는 방식으로 진행한다.
- **가장 큰 리스크는 git 충돌이 아니라 계약 불일치**: §4를 구현하는 에이전트(앱 쪽)와
  `workbook_review_apply.py`를 확장하는 에이전트(agent-packages 쪽)가 서로 다른
  브랜치에서 독립적으로 "완료"를 선언할 수 있다 — 위 §4의 "미해결 간극"이 해소되지
  않은 채로. 이건 리뷰 없이는 드러나지 않는다. §4와 apply 확장은 같은 주체가 하거나,
  manifest 변환 방식을 먼저 확정한 뒤에만 각자 브랜치를 열 것을 권장한다.
- **§5는 §4 뒤에 온다**: `/st-review`가 만드는 리포트·제안 manifest는 `report_format_spec.md`
  계약을 따르므로, §4가 그 형식을 확정하기 전에 §5를 구현하면 명령 인터페이스를 다시
  고쳐야 한다. 또 §5는 `commands/` 전체와 `SKILL.md`를 광범위하게 건드려 다른 에이전트
  브랜치의 문서 수정과 충돌하기 쉬우므로, 명령 통합 작업 중에는 `commands/` 편집을
  한 브랜치로 제한한다.
- **§6은 §5 뒤에 온다**: ChatGPT for Excel 레이어는 `/st-edit`의 preview·`before` 검증·
  diff 검증 계약을 재사용하는 전제이므로, §5가 그 계약을 확정하기 전에는 PoC를
  시작하지 않는다. 반면 §6은 `excel-chatgpt/` 신규 디렉터리 위주라 일단 시작하면
  기존 파일과의 충돌 위험은 낮다.
- **계획 문서 자체의 충돌**: 이 로드맵과 `roadmap_obsidian_agent_workflow.md`,
  `report_format_spec.md`를 여러 에이전트가 동시에 고치면 서로의 최신 결정을 덮어쓸
  수 있다. 계획 문서 편집은 한 번에 한 에이전트만 하고, 구현 브랜치는 머지된 최신
  버전만 참조한다.

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
- `Spanish`/`ES`는 Spain Spanish 규칙을 유지하고, `Spanish_Colombia`/`CO(콜롬비아)`는 `es_CO`,
  기본 `tú`, vos 금지, 라틴아메리카 어휘 규칙을 선택한다.
- CO의 RAG 조회는 `es_CO` 사례를 ES보다 우선하며, `target_lang` 정확 일치로 ES와 격리된다.
- 정상 규칙 파일은 프롬프트에 반영되고, 누락 필드·중복 키·잘못된 schema version은 테스트 가능한 오류를 낸다.
- 에이전트 검수 전용 규칙도 `scope`와 locale로 검색할 수 있으며, glossary·명시 규칙·RAG의
  우선순위가 일관되게 적용된다.
- 생성되는 검수 리포트의 확장자와 API MIME type이 Markdown이며, ZIP에도 `.md` 리포트와
  기계 판독 가능한 수정 manifest가 포함된다.
- 각 수정 제안은 시트/셀·근거 규칙·적용 전후 값을 식별할 수 있고, 승인되지 않은 제안은 Excel에 반영되지 않는다.
- `/st-edit`는 승인 전 preview와 `before` 검증을 수행하고, 원본이 아닌 새 파일에만
  적용한다. `draft`와 재하이라이트·검증을 마친 `delivery` 산출물을 구분한다.
- ChatGPT for Excel 레이어는 공통 규칙과 manifest 계약을 재사용하며, 지원하지 않는
  Office.js 기능 또는 rich text 보존 불확실성이 있으면 수정하지 않고 CLI 경로로 fallback한다.
- glossary 하이라이팅은 Live Excel preview 경로와 Delivery Python rich text 경로를 모두
  제공하며, 부분 문자열 하이라이트의 납품 기준은 Delivery Python 검증을 통과한 결과다.
- `/viewer`에서 Markdown 파일 업로드와 API URL 자동 로드가 가능하며, 비신뢰 HTML은 정화된다.
