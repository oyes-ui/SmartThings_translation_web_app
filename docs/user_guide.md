# SmartThings Translation Web App 통합 가이드

이 문서는 웹 앱의 설치·실행과 기본 사용을 안내합니다. 번역 에이전트는
[업무 가이드](../agent-packages/smartthings-translation-agent/references/workflow-guide.md)와
[기본 명령](../agent-packages/smartthings-translation-agent/commands/README.md)을 함께 확인하세요.
웹 앱의 API 실행과 에이전트의 읽기 전용 검수·승인 납품은 별도 경로입니다.

---

## 1. 설치 및 시작하기 (Setup & Execution)

### 1.1 사전 준비 사항 (Prerequisites)
시작하기 전에 다음 항목들이 준비되었는지 확인하세요:
- **Python 3.10 이상**: [python.org](https://www.python.org/)에서 설치 가능합니다.
- **Google API Key**: Gemini 모델 사용을 위해 필요합니다. [Google AI Studio](https://aistudio.google.com/)에서 발급받으세요.
- **OpenAI API Key** (선택 사항): 앱에서 OpenAI 검수 모델을 선택할 경우 필요합니다.

### 1.2 환경 설정 (Environment Setup)
1. **저장소 이동**: 터미널을 열고 프로젝트 폴더로 이동합니다.
2. **가상환경 생성 및 활성화**:
   - **Mac/Linux**: `python3 -m venv venv && source venv/bin/activate`
   - **Windows**: `python -m venv venv && .\venv\Scripts\activate`
3. **패키지 설치**: `pip install -r requirements.txt`

### 1.3 환경 변수 설정 (.env)
프로젝트 루트에 `.env` 파일을 생성하고 API 키를 입력합니다.
```env
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 1.4 애플리케이션 실행 (Running the App)
- **Mac/Linux**: `bash run_mac.command`
- **Windows**: `go-webui.bat`
- **수동 실행**: `PYTHONPATH=src uvicorn translation_web_app.main:app --reload --reload-dir src --host 0.0.0.0 --port 8000`

실행 후 브라우저에서 `http://localhost:8000`에 접속하세요.

---

## 2. 기본 검수 워크플로우

1. **파일 업로드**: 검수할 엑셀(`.xlsx`)과 용어집(`.csv`, 선택)을 업로드합니다.
2. **시트 매핑**: 원본 시트와 타겟 시트를 지정합니다. 시스템이 자동으로 매핑을 제안하며, 필요시 JSON 에디터에서 `lang`과 `code`를 직접 수정할 수 있습니다.
3. **구성 설정**: 사용할 모델, BX 스타일 적용 여부, RAG 사용 여부 등을 선택합니다.
   - **API Key**: `.env` 설정이 없어도 화면 하단 'API Keys' 패널에서 세션별 키를 임시 저장하여 사용할 수 있습니다.
4. **검수 시작**: [Start Inspection] 버튼을 눌러 검수를 진행하고 실시간 로그를 확인합니다.
5. **결과 다운로드**: 완료 후 생성된 통합 엑셀 파일과 검수 리포트를 ZIP으로 다운로드하거나, HTML 리포트 뷰어로 즉시 확인합니다.

---

## 3. 용어집(Glossary) 활용 A to Z

시스템은 문맥(`row_key`)을 분석하여 용어집을 지능적으로 적용합니다.

### 3.1 대괄호(Bracket) 자동화 규칙
- **자동 적용 (Description/Body)**: `row_key`가 설명문 계열일 때 용어에 `[]` 또는 `「」`를 붙입니다.
- **자동 제외 (Title/Button)**: 제목·소제목·버튼의 독립 용어는 괄호·따옴표 없이 표시하며 용어집 대소문자는 유지합니다. 명시적으로 인용한 UI 경로는 별도로 판단합니다.
- **수동 제외 (Exempt Markers)**: 용어집 `Rule`의 `no bracket`·`대괄호 제외` 등 표식을 확인합니다. 비활성 용어와 셀별 활성화 예외는 확정한 작업 범위를 따릅니다.

### 3.2 내비게이션 경로 특수 규칙
“설정 > 기기” 같은 경로는 대상 언어에 맞는 인용부호로 감싸며 경로 내부 용어에는 별도 괄호를
붙이지 않습니다. 일반 경로의 문장 끝 마침표는 닫는 인용부호 밖에 둡니다. 동아시아 언어별
세부 표기는 [런타임 용어 규칙](../src/translation_web_app/rules/glossary.md)과
[언어 규칙](../src/translation_web_app/rules/languages)을 확인합니다.

---

## 4. 핵심 기능 상세

### 4.1 하이브리드 RAG (유사 사례 참조)
- **Identity Match**: 같은 원문의 과거 번역을 우선 참고합니다. 과거 사례가 현재 용어집·문법·기능 조건을 덮어쓰지는 않습니다.
- **Semantic Match**: 설정한 범위에서 의미적으로 유사한 사례를 참고 자료로 제공합니다. 쿼리 임베딩에는 API 비용이 발생할 수 있습니다.
- **RAG DB Viewer**: 상단 메뉴를 통해 현재 DB에 저장된 모든 번역 쌍을 필터링하여 조회할 수 있습니다.
- **DB 위치**: 기본 저장 위치는 `runtime/rag_db/`입니다. 기존 `rag_db/` 데이터가 있다면 재빌드하지 않고 `rag_store.db`와 `chroma/`를 새 위치로 이관해 사용할 수 있습니다.

### 4.2 BX Style Transcreation (삼성 브랜드 보이스)
삼성의 브랜드 아이덴티티(**OPEN, BOLD, AUTHENTIC**)를 반영하여 단순 번역을 넘어선 트랜스크리에이션을 수행합니다. 전용 페르소나('자신감 있는 탐험가')가 활성화됩니다.

### 4.3 언어별 특화 규칙 (Localization Rules)
- **독일어**: `Du-form` 기본 사용
- **일본어**: `ます형` 기본 사용 및 내비게이션 경로 `「 」` 적용
- **포르투갈어/중국어**: 언어 규칙과 용어집 컬럼을 구분합니다. BR은 브라질 언어 규칙과 포르투갈 용어집 컬럼을 기본으로 사용하며, 간체·번체도 대상 언어에 맞게 설정합니다.

---

## 5. 에이전트로 번역·검수·납품하기

1. 패키지 폴더에서 `scripts/bootstrap.py --app-root <앱 경로> --json`으로 연결 상태를 확인합니다.
2. `/st-translate`는 소량 직접 번역이 기본입니다. 대량 API 번역은 비용 승인 범위에서 실행합니다.
3. `/st-inspect`는 셀 순차 검수 → 시트 일관성 검수 → 리드 통합을 수행하고 승인 대기 제안을 만듭니다.
4. 지정 문구의 draft는 `/st-edit`, 검수 승인안의 최종 납품은 `/st-apply`로 만듭니다.
5. 원본은 보존합니다. 반환된 버전 경로와 독립 검증 결과를 확인하며 draft를 납품본으로 안내하지 않습니다.

에이전트의 공용 저장·독립 검증은 공백·줄바꿈·글꼴·색과 승인된 변경 범위를 확인합니다.
웹 앱의 독립 실행 결과나 외부에서 만든 파일에 같은 보장이 자동 적용되는 것은 아닙니다.
상세한 실행·재개 기준은 [실행 규약](../agent-packages/smartthings-translation-agent/references/command-execution.md)을 따릅니다.

### 업무 종료와 작은 교정 기록

종료 시 현재 작업의 확정 판단·실행 검증·기존 outcome 경로를 확인합니다. 오탈자·도구 인자·
경로·중복 질문 같은 작은 오류도 현재 승인 범위에서 교정하고 기존 리포트나 완료 답변에 남깁니다.
“이번 작업에서 개선할 점 찾아줘”라고 요청하면 기록을 분석하며 공통 절차·규칙 변경과 효과 확인은
요청 범위에서 수행합니다. 별도 회고 명령이나 백그라운드 자동 분석은 없습니다.
[최소 개선 루프](../agent-packages/smartthings-translation-agent/references/improvement-loop.md)를 참고하세요.

## 6. 디버깅 및 분석 도구

### 6.1 Prompt Inspector
`demo.html`에서 언어와 `row_key`에 따라 실시간으로 조립되는 프롬프트 블록을 확인하고 테스트할 수 있습니다.

### 6.2 Prompt Universe
`universe.html`에서 시스템의 모든 프롬프트 모듈(Common, Language, BX, Context 등)이 어떻게 연결되어 있는지 그래프 형태로 한눈에 파악할 수 있습니다.

---

> [!INFO]
> 실행 규칙의 정본은 [rules/](../src/translation_web_app/rules/)입니다.
> [종합 규칙 모음](comprehensive_rules.md)은 설명 문서이며 충돌 시 런타임 규칙을 우선합니다.
