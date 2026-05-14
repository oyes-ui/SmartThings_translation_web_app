---
title: SmartThings Translation Web App
emoji: "✨"
colorFrom: "blue"
colorTo: "purple"
sdk: "docker"
app_port: 7860
pinned: false
---

# ✨ SmartThings Translation Web App

**Gemini Pro & GPT-5.4(Audit)를 활용한 다국어 번역 정합성 검수 자동화 도구**

이 웹 애플리케이션은 다국어 엑셀 파일(.xlsx)을 분석하여 번역의 일관성, 용어집(Glossary) 준수 여부, 문맥 상의 오류를 AI로 진단하고 리포트를 생성합니다.

---

## 🚀 주요 기능 (Key Features)

- **AI 기반 정밀 검수**: Gemini 3.1/3.0 및 GPT-5.4 모델을 포함한 고성능 번역 검수 지원
- **하이브리드 RAG (Retrieval-Augmented Generation)**: 100% 일치(Identity Match) 및 의미적 유사도 검색을 통한 번역 일관성 확보
- **다국어 시트 자동 매핑**: 국가 코드(US, KR, DE 등)를 인식하여 시트별 언어 설정 자동화
- **용어집(Glossary) 검증**: CSV 용어집 기반 정밀 매칭 및 괄호(Bracket) 자동 처리 규칙 적용
- **V2 Localization Engine**: 브라질/유럽 포르투갈어 및 중국 간체/번체 시장별 완벽 분리 대응
- **Context-Aware Formatting**: Row Key 기반으로 타이틀/버튼 vs 설명문 맥락을 자동 판정하여 용어집 괄호 처리 최적화
- **Typography Standards**: 언어별 표준 문장 부호 및 타이포그래피 규칙 자동 적용
- **Prompt Inspector**: 실시간 프롬프트 조립 상태를 육안으로 확인할 수 있는 전용 디버그 도구 제공

---

## 📚 튜토리얼 및 규칙 가이드 (Guides & Rules)

시스템 사용법부터 내부 번역 규칙까지 단계별로 안내합니다.

1.  **[가이드 01] 통합 사용 가이드**: [user_guide.md](docs/user_guide.md) - 설치, 실행, 워크플로우 및 용어집 활용 (A to Z)
2.  **[가이드 02] 종합 규칙 모음**: [comprehensive_rules.md](docs/comprehensive_rules.md) - 프롬프트 로직, 국가별 규칙, BX 스타일 등 모든 내부 엔진 규칙 참조

---

## 🏗️ 프롬프트 아키텍처 (Prompt Architecture)

이 프로젝트는 고도로 모듈화된 프롬프트 시스템을 사용합니다. 상세 구조는 [Prompt Architecture Document](docs/prompt_architecture.html)에서 확인할 수 있습니다.

- **Persona_and_Task**: 기본 번역가 페르소나 설정
- **Samsung_BX_Guidelines**: 브랜드 가이드라인 동적 주입
- **Language_Specific_Hints**: 언어별 문법/문화적 뉘앙스 처리
- **Typography_Rules**: 타겟 언어별 문장 부호 및 타이포그래피 규칙
- **RAG_Reference**: 유사 사례 참조 (Identity/Semantic)
- **Context_Routing**: `row_key` 판정을 통한 컨텍스트별 가이드라인 자동 분기

---

## 🛠️ 사전 설정 (Hugging Face Spaces)

이 프로젝트를 Hugging Face Spaces에서 실행하려면, **Secret Key** 설정이 필수입니다.

1. Spaces 대시보드의 **Settings** 탭으로 이동합니다.
2. **Variables and Secrets** 섹션에서 `New Secret`을 클릭합니다.
3. 아래 키를 등록합니다:
    - `GOOGLE_API_KEY`: Google AI Studio에서 발급받은 Gemini API Key

> **Note**: 기본 포트는 `7860`으로 설정되어 있습니다.

---

## 💻 로컬 실행 방법 (Local Development)

### 1. 환경 설정

Python 3.10 이상이 필요합니다.

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 파일 생성 (.env)

프로젝트 루트에 `.env` 파일을 생성하고 API 키를 입력하세요.

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. 애플리케이션 실행

**방법 A: 실행 스크립트 사용 (권장)**
- **Mac/Linux**: `bash run_mac.command`
- **Windows**: `go-webui.bat` 실행

**방법 B: 수동 실행**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
브라우저에서 `http://localhost:8000`으로 접속합니다. (Prompt Inspector: `/demo.html`)

---

## 🐳 Docker 실행

이미지가 빌드된 후 7860 포트로 실행됩니다.

```bash
# 이미지 빌드
docker build -t translation-checker .

# 컨테이너 실행
docker run -p 7860:7860 -e GOOGLE_API_KEY="your_key_here" translation-checker
```

---

---

## 📦 Release History (최근 주요 업데이트)

상세한 변경 내역은 [CHANGELOG.md](docs/CHANGELOG.md)에서 확인할 수 있습니다.

### [v1.5.0] - 2026-05-12
- **Enhanced**: Localization Engine V2 업데이트 (포르투갈어/중국어 시장별 분리)
- **Added**: Context-Aware Glossary 처리 로직 (Title/Button vs Description)
- **Added**: Prompt Inspector 디버그 페이지 추가 (`/static/demo.html`)
- **Added**: 언어별 타이포그래피 및 문장 부호 표준화 규칙 적용
- **Refactored**: `PromptBuilder` 및 `prompt_modules` 구조적 리팩토링

### [v1.4.0] - 2026-04-15
- **Fixed**: `checker_service.py` 구문 오류(SyntaxError) 수정 및 앱 실행 불가 이슈 해결
- **Added**: 일반 채팅(ChatGPT, Gemini)용 프롬프트 마스터 문서 추가 (`docs/prompts_for_chat.md`)

---

## 📝 사용 가이드 (User Guide)

1. **Configurations (좌측 패널)**: 검수 모드, 사용할 AI 모델(Gemini/GPT), BX 스타일 적용 여부를 설정합니다.
2. **Files & Execution (중앙 패널)**: 
    - **Upload Workbook**: 번역할 엑셀 파일(.xlsx)을 업로드합니다.
    - **Glossary (Optional)**: 용어집(.csv)을 업로드합니다.
    - **Sheet Mapping**: 업로드 후 생성된 시트 리스트에서 원본 및 타겟 시트를 선택합니다.
    - **RAG DB**: 필요한 경우 RAG DB를 빌드하거나 특정 스토리 데이터를 업데이트합니다.
3. **Start Inspection**: 설정을 검증([Validate Config])한 후 검수를 시작합니다.
4. **Live Progress & Download (우측 패널)**: 터미널에서 진행 상황을 확인하고, 완료 시 리포트를 다운로드하거나 HTML 뷰어로 확인합니다.
5. **Debug**: `http://localhost:8000/demo.html`에서 프롬프트 조립 상태를 실시간 점검할 수 있습니다.