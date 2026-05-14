# SmartThings Translation Web App 통합 가이드

이 문서는 **SmartThings Translation Web App**의 설치부터 핵심 기능 활용, 용어집 적용 규칙까지 모든 과정을 담은 통합 가이드입니다.

---

## 1. 설치 및 시작하기 (Setup & Execution)

### 1.1 사전 준비 사항 (Prerequisites)
시작하기 전에 다음 항목들이 준비되었는지 확인하세요:
- **Python 3.10 이상**: [python.org](https://www.python.org/)에서 설치 가능합니다.
- **Google API Key**: Gemini 모델 사용을 위해 필요합니다. [Google AI Studio](https://aistudio.google.com/)에서 발급받으세요.
- **OpenAI API Key** (선택 사항): GPT-4/5 계열 모델로 감수를 진행할 경우 필요합니다.

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
- **수동 실행**: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

실행 후 브라우저에서 `http://localhost:8000`에 접속하세요.

---

## 2. 기본 검수 워크플로우

1. **파일 업로드**: 검수할 엑셀(`.xlsx`)과 용어집(`.csv`, 선택)을 업로드합니다.
2. **시트 매핑**: 원본 시트와 타겟 시트를 지정합니다. 시스템이 자동으로 매핑을 제안합니다.
3. **구성 설정**: 사용할 모델, BX 스타일 적용 여부, RAG 사용 여부 등을 선택합니다.
4. **검수 시작**: [Start Inspection] 버튼을 눌러 검수를 진행하고 실시간 로그를 확인합니다.
5. **결과 다운로드**: 완료 후 생성된 통합 엑셀 파일과 검수 리포트를 ZIP으로 다운로드합니다.

---

## 3. 용어집(Glossary) 활용 A to Z

시스템은 문맥(`row_key`)을 분석하여 용어집을 지능적으로 적용합니다.

### 3.1 대괄호(Bracket) 자동화 규칙
- **자동 적용 (Description/Body)**: `row_key`가 설명문 계열일 때 용어에 `[]` 또는 `「」`를 붙입니다.
- **자동 제외 (Title/Button)**: 버튼명이나 제목에는 가독성을 위해 대괄호를 붙이지 않습니다.
- **수동 제외 (Exempt Markers)**: 용어집의 remark 등에 `no bracket` 혹은 `대괄호 제외` 키워드가 있으면 무조건 제외합니다.

### 3.2 내비게이션 경로 특수 규칙
"설정 > 기기"와 같은 경로는 **이중 따옴표(`" "`)**로 감싸며, 내부 용어에는 대괄호를 붙이지 않습니다. 마침표 위치는 언어별 표준(US는 따옴표 밖, 그 외는 안)을 따릅니다.

---

## 4. 핵심 기능 상세

### 4.1 하이브리드 RAG (유사 사례 참조)
- **Identity Match**: 100% 일치하는 과거 번역을 최우선 적용합니다.
- **Semantic Match**: 의미적으로 유사한 사례를 AI에게 참고 자료로 제공합니다.

### 4.2 BX Style Transcreation (삼성 브랜드 보이스)
삼성의 브랜드 아이덴티티(**OPEN, BOLD, AUTHENTIC**)를 반영하여 단순 번역을 넘어선 트랜스크리에이션을 수행합니다.

### 4.3 언어별 특화 규칙 (Localization Rules)
- **독일어**: `Du-form` 기본 사용
- **일본어**: `ます형` 기본 사용
- **포르투갈어/중국어**: 시장별(브라질vs유럽, 간체vs번체) 용어 체계 완벽 분리

---

## 5. 디버깅 및 도구

### Prompt Inspector
`http://localhost:8000/static/demo.html`에서 언어와 `row_key`에 따라 실시간으로 조립되는 프롬프트를 확인할 수 있습니다. 대괄호 규칙이나 프롬프트 구성을 테스트할 때 유용합니다.

---

> [!INFO]
> 모든 국가별 상세 규칙과 프롬프트 로직은 **[[가이드 02] 종합 규칙 모음](./comprehensive_rules.md)**에서 확인할 수 있습니다.
