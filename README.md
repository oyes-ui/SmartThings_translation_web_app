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

**다국어 Excel 번역·검수 웹 앱과 SmartThings 번역 에이전트 패키지**

이 웹 애플리케이션은 다국어 엑셀 파일(.xlsx)을 분석하여 번역의 일관성, 용어집(Glossary) 준수 여부, 문맥 상의 오류를 AI로 진단하고 리포트를 생성합니다.

---

## 🚀 주요 기능 (Key Features)

- **AI 기반 정밀 검수**: Gemini 3.8 Flash / 3.1 Pro 및 GPT-5.6/5.5/5.4 계열을 포함한 고성능 번역 검수 지원 (기본 번역 모델: `gemini-3.8-flash`)
- **하이브리드 RAG (Retrieval-Augmented Generation)**: 100% 일치(Identity Match) 및 의미적 유사도 검색을 통한 번역 일관성 확보
- **다국어 시트 자동 매핑**: 국가 코드(US, KR, DE 등)를 인식하여 시트별 언어 설정 자동화
- **다중 소스 그룹 처리**: KR/US 등 여러 원문 시트를 각 타겟 시트 그룹에 매핑하여 검수, 번역+검수, 하이라이트 전용 작업을 한 번에 실행
- **용어집(Glossary) 검증**: CSV 용어집 기반 정밀 매칭 및 괄호(Bracket) 자동 처리 규칙 적용
- **시장별 현지화**: 언어별 규칙과 용어집 매핑을 구분. BR은 브라질 언어 규칙을 유지하며 기본 용어집은 포르투갈 컬럼을 공유
- **Context-Aware Formatting**: Row Key 기반으로 타이틀/버튼 vs 설명문 맥락을 자동 판정하여 용어집 괄호 처리 최적화
- **Typography Standards**: 언어별 표준 문장 부호 및 타이포그래피 규칙 자동 적용
- **Prompt Inspector**: 실시간 프롬프트 조립 상태를 육안으로 확인할 수 있는 전용 디버그 도구 제공
- **에이전트 Excel 수정·납품**: 승인 범위를 계약에 고정하고 공용 atomic 저장·독립 검증을 거친 버전 사본을 공개. 공백·줄바꿈·문자별 글꼴·색을 검증
- **시트 단위 에이전트 검수**: 셀 순차 검수 → 시트 일관성 검수 → 리드 통합이 기본. 제안은 읽기 전용이며 5역할 검수는 deprecated 호환 경로
- **업무 종료·개선 루프**: 기존 outcome과 실행 결과를 확인하고 작은 교정을 기록. 원인 분석·공통 절차 변경·효과 평가는 요청 범위에서 수행
- **Excel Live 레이어**: ChatGPT for Excel과 Claude for Excel에서 같은 승인 manifest·fallback 정책을 공유

---

## 📚 튜토리얼 및 규칙 가이드 (Guides & Rules)

시스템 사용법부터 내부 번역 규칙까지 단계별로 안내합니다.

1.  **[가이드 01] 통합 사용 가이드**: [user_guide.md](docs/user_guide.md) - 설치, 실행, 워크플로우 및 용어집 활용 (A to Z)
2.  **[가이드 02] 종합 규칙 모음**: [comprehensive_rules.md](docs/comprehensive_rules.md) - 프롬프트 로직, 국가별 규칙, BX 스타일 등 모든 내부 엔진 규칙 참조

### 번역 에이전트 시작하기

사용자 명령은 [명령 안내](agent-packages/smartthings-translation-agent/commands/README.md)의
여섯 개가 기본입니다. 파일·언어·하고 싶은 일을 자연어로 전달하면 에이전트가 내부 설정을 준비합니다.

| 명령 | 용도 |
| --- | --- |
| `/st-start` | 연결 상태·현재 작업·다음 단계 |
| `/st-ask` | 규칙·용어·사례 질문과 업무 후 개선 분석 |
| `/st-translate` | 소량 직접 번역 또는 승인한 대량 API 번역 |
| `/st-inspect` | 셀→시트→리드 검수와 승인 대기 제안 |
| `/st-edit` | 지정한 내용의 수정본(draft) 생성 |
| `/st-apply` | 승인안을 반영·하이라이트·검증한 납품본 생성 |

번역 에이전트는 **패키지 폴더에서** 시작하고 앱 저장소를 연결합니다.

```bash
cd agent-packages/smartthings-translation-agent
python scripts/bootstrap.py --app-root ../.. --json
# 같은 앱 연결을 이후에도 사용할 때
python scripts/bootstrap.py --app-root ../.. --save
```

소량 작업은 별도 번역·검수 API를 호출하지 않는 셀프 모드가 기본입니다. API 번역·검수 및
semantic RAG는 비용·범위 승인 후 사용합니다. `/st-pipeline`은 호환 명령이며 새 업무는
`/st-translate`·`/st-inspect`에서 시작합니다.

`/st-inspect`의 제안은 `pending_approval`로 남고 사람 승인 뒤 반영합니다. 계약 실행기가 반환한
`revised`는 draft, 납품의 `final`은 `verified/<work_id>/` 안의 검증된 배치입니다.
파일 존재만으로 납품 완료로 보지 않으며 이 보장을 웹 앱의 모든 독립 저장 경로에 확대하지 않습니다.

- [업무 가이드](agent-packages/smartthings-translation-agent/references/workflow-guide.md): 업무별 입력·진행·지원 범위
- [실행·재개 규약](agent-packages/smartthings-translation-agent/references/command-execution.md): 기존 승인 재사용과 검증된 배치
- [업무 종료 후 개선 루프](agent-packages/smartthings-translation-agent/references/improvement-loop.md): 자동 기록과 요청 기반 개선
- [설치·이식성](agent-packages/smartthings-translation-agent/references/portability.md): 공통 본체와 도구별 진입점

“1차 개선 모드”의 전체 자동 연결은 아직 지원 완료가 아닙니다. 개별 검수·승인 반영 경로와
[현재 지원 경계](agent-packages/smartthings-translation-agent/references/workflow-guide.md#1차-개선-모드)를 확인합니다.

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
PYTHONPATH=src uvicorn translation_web_app.main:app --reload --reload-dir src --host 0.0.0.0 --port 8000
```
브라우저에서 `http://localhost:8000`으로 접속합니다. (Prompt Inspector: `/demo.html`)

---

## 🧠 RAG DB 저장 위치

RAG DB의 기본 저장 위치는 `runtime/rag_db/`입니다. 기존 개발 환경의 `rag_db/`에 이미 구축된 DB가 있다면 API 비용이 드는 재빌드 대신 `rag_store.db`와 `chroma/`를 `runtime/rag_db/`로 이관하면 됩니다.

```bash
PYTHONPATH=src python -m translation_web_app.rag_db_builder --status
```

`RAG_DB_DIR` 환경변수를 지정하면 임시 테스트나 별도 배포 환경에서 다른 DB 위치를 사용할 수 있습니다.

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

### [Unreleased]

- 기본 명령 6개와 셀→시트→리드 검수, 여러 워크북의 작업별 설정·승인표 흐름을 정리했습니다.
- 공용 저장의 텍스트·문자 서식 보호와 납품·재개의 독립 검증을 강화했습니다.
- 검수 셀 범위 지정, 일본어 괄호 판정, BR 용어집 매핑과 저장된 번역의 오류·빈 셀 검증을 보강했습니다.
- 기존 outcome을 활용한 종료 절차와 작은 번역·진행 오류 기록, 요청 기반 개선 분석을 추가했습니다.
- 기본 번역 모델과 재시도 처리의 변경 내역은 [Changelog](docs/CHANGELOG.md#unreleased)를 참고합니다.

### [v1.7.0] - 2026-06-25
- **Added**: 번역 에이전트 셀프 모드 — `prompt_preview.py` 가 앱과 동일한 프롬프트를 조립해 LLM 크레딧 없이 직접 번역/검수
- **Added**: 에이전트 래퍼 — 용어집 관리, 텍스트워크북 생성, 번역/검수 파이프라인(`--pipeline` 가드)
- **Added**: 슬래시 명령어 13종(`/st-*`)과 프로젝트 안내(`/st-help`)
- **Safety**: 용어집 쓰기 `--apply` 가드, 리포트 atomic write, 미완료 실행 `incomplete` 처리

### [v1.6.0] - 2026-06-02
- **Added**: 다중 소스 그룹(`source_groups`) 기반 통합 번역+검수 파이프라인 지원
- **Added**: 하이라이트 전용 파이프라인의 다중 소스 그룹 처리 지원
- **Changed**: 검수/번역/하이라이트 파이프라인의 멀티소스 진입 조건을 `source_groups` 기준으로 통일
- **Changed**: RAG DB 기본 경로를 `runtime/rag_db/`로 정리하고 기존 `rag_db/` 데이터는 이관 방식으로 유지
- **Fixed**: 하이라이트 전용 다중 소스 경로에서 `asyncio` 로컬 import로 발생하던 `UnboundLocalError` 수정
- **Verified**: 컴파일, 기존 프롬프트 유닛 테스트, 다중 소스 번역/하이라이트 스모크 테스트 통과

### [v1.5.0] - 2026-05-12
- **Enhanced**: Localization Engine V2 업데이트 (포르투갈어/중국어 시장별 분리)
- **Added**: Context-Aware Glossary 처리 로직 (Title/Button vs Description)
- **Added**: Prompt Inspector 디버그 페이지 추가 (`/demo.html`)
- **Added**: 언어별 타이포그래피 및 문장 부호 표준화 규칙 적용
- **Refactored**: `PromptBuilder` 및 `prompt_modules` 구조적 리팩토링

### [v1.4.0] - 2026-04-15
- **Fixed**: `src/translation_web_app/checker_service.py` 구문 오류(SyntaxError) 수정 및 앱 실행 불가 이슈 해결
- **Added**: 일반 채팅(ChatGPT, Gemini)용 프롬프트 마스터 문서 추가 (`docs/prompts_for_chat.md`)

---

## 📝 사용 가이드 (User Guide)

1. **Configurations (좌측 패널)**: 검수 모드, 사용할 AI 모델(Gemini/GPT), BX 스타일 적용 여부를 설정합니다.
2. **Files & Execution (중앙 패널)**: 
    - **Upload Workbook**: 번역할 엑셀 파일(.xlsx)을 업로드합니다.
    - **Glossary (Optional)**: 용어집(.csv)을 업로드합니다.
    - **Sheet Mapping**: 업로드 후 생성된 시트 리스트에서 원본 및 타겟 시트를 선택합니다. 필요한 경우 여러 원본 시트 그룹을 구성해 각 그룹별 타겟 시트를 따로 지정할 수 있습니다.
    - **RAG DB**: 필요한 경우 RAG DB를 빌드하거나 특정 스토리 데이터를 업데이트합니다.
3. **Start Inspection**: 설정을 검증([Validate Config])한 후 검수를 시작합니다.
4. **Live Progress & Download (우측 패널)**: 터미널에서 진행 상황을 확인하고, 완료 시 리포트를 다운로드하거나 HTML 뷰어로 확인합니다.
5. **Debug**: `http://localhost:8000/demo.html`에서 프롬프트 조립 상태를 실시간 점검할 수 있습니다.
