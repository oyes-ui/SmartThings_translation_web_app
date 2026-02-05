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

**Gemini Pro & GPT-4o(Audit)를 활용한 다국어 번역 정합성 검수 자동화 도구**

이 웹 애플리케이션은 다국어 엑셀 파일(.xlsx)을 분석하여 번역의 일관성, 용어집(Glossary) 준수 여부, 문맥 상의 오류를 AI로 진단하고 리포트를 생성합니다.

---

## 🚀 주요 기능 (Key Features)

- **AI 기반 정밀 검수**: Gemini 2.5/2.0 Pro 및 GPT-4o 모델을 활용한 고성능 번역 검수
- **다국어 시트 자동 매핑**: 국가 코드(US, KR, DE 등)를 인식하여 언어별 시트 자동 처리
- **용어집(Glossary) 검증**: 업로드된 CSV 용어집을 기반으로 필수 용어 준수 여부 확인
- **Samsung BX Style Transcreation**: 'Persona' 및 'Voice & Tone' 설정을 통한 브랜드 톤앤매너 검수
- **실시간 로그 & 리포트**: 웹 터미널을 통한 진행 상황 모니터링 및 결과 파일(.txt/.zip) 다운로드

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
브라우저에서 `http://localhost:8000`으로 접속합니다.

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

## 📝 사용 가이드 (User Guide)

1. **Upload Workbook**: 번역 검수를 진행할 원본 엑셀 파일(.xlsx)을 업로드합니다.
2. **Configuration**: 
    - **AI Model**: 검수에 사용할 모델(Gemini 2.5 Pro 등)을 선택합니다.
    - **Sheet Mapping**: 시트 이름과 언어 코드가 올바르게 매핑되어 있는지 JSON 설정을 확인합니다.
3. **Glossary (Optional)**: 용어집(.csv)이 있다면 업로드하여 정확도를 높입니다.
4. **Start Inspection**: [Start Inspection] 버튼을 눌러 검수를 시작합니다.
5. **Download**: 우측 패널에서 진행 상황을 확인하고, 완료 시 결과 리포트를 다운로드합니다.