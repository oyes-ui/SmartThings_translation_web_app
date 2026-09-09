# Changelog - SmartThings Translation Checker

## [Unreleased]

### Added
- **Gemini 3.8 지원**: 번역 모델·검수 모델 셀렉트에 `Gemini 3.8 Series`(Thinking On/Off) 추가. Vertex AI `publishers/google/models/gemini-3.8-flash` 로 확인된 모델 ID 사용.
- **`model_pricing.py` (단가 단일 출처)**: 공식 페이지(2026-09-09 확인) 기준 Gemini/OpenAI standard tier 단가를 한 곳에 모음. Gemini 3.6~3.8 Flash 의 도입가(2026-12-31 만료)를 실행 날짜로 자동 전환해, 2027-01-01 이후 추정치가 조용히 절반으로 어긋나지 않게 함.

### Removed
- **Vertex 에 없는 죽은 옵션 제거**: `gemini-3-pro-preview`, `gemini-3.1-flash-lite-preview` 는 이 프로젝트 Vertex 모델 목록에 존재하지 않아 선택 시 호출 단계에서 실패하던 옵션이었음(목록에는 `-preview` 접미사 없는 `gemini-3.1-flash-lite` 만 존재).

### Fixed
- **비용 추정 단가 오류**: `co_batch_cost_estimate.py` 의 3rd-party aggregator 참고치를 공식값으로 교체. Gemini 3.6 Flash 는 도입가가 아닌 표준가(1.50/7.50)를 써서 현재 시점 기준 약 2배 과다 계상, `gpt-5.2` 는 standard(1.75/14.00) 대신 batch 단가(0.875/7.00)를 써서 약 절반 과소 계상되고 있었음.

### Changed
- **기본 번역 모델 `gemini-3.6-flash` → `gemini-3.8-flash`**: `ModelHandler.call_gemini`/`generate_content`/`count_tokens`, `TranslationChecker._run_llm_translation`/`run_integrated_pipeline_generator`, `main.StartRequest`, `/api/preview_prompt_blocks`, `TextWorkbookStartRequest` 기본값 일괄 갱신. UI 기본 선택도 Gemini 3.8 Flash (Thinking: On).
- **Gemini 모델 셀렉트 정리**: Flash 는 3.8 하나만, Pro 는 최신(`gemini-3.1-pro-preview`) 하나만 남기고 3.6/3.5/3.1-flash-lite/3.0-preview/2.5 시리즈를 번역·검수 셀렉트 양쪽에서 제거. optgroup 도 버전 나열 대신 `Gemini Flash` / `Gemini Pro` 로 단순화. GPT 계열은 변경 없음.
- **agent-package 기본 모델**: `workbook_translate.py`·`batch_co_rollout.py` 의 `--translation-model` 기본값과 `co_batch_cost_estimate.py` 의 `TRANSLATION_MODEL` 을 `gemini-3.8-flash` 로 상향.
- `demo.html` 이 `/api/preview_prompt_blocks` 에 하드코딩하던 `gemini-2.5-flash` 를 제거하고 엔드포인트 기본값을 따르게 함(기본값 이중 관리 해소).
- **비용 추정 스크립트 단가 참조 일원화**: `scripts/token_cost_report.py`(`gemini-3-flash-preview` → `gemini-3.8-flash`), `scripts/token_cost_report_cell_rag.py`(`gemini-3.5-flash` → `gemini-3.8-flash`), `agent-packages/.../co_batch_cost_estimate.py` 가 각자 들고 있던 PRICING 리터럴을 `model_pricing` 참조로 교체. `token_cost_report_cell_rag.py` 의 `--translation-model`/`--audit-model` 선택지도 전체 모델로 확대.

---

## [1.7.0] - 2026-06-25

### Added (Agent Skill — `agent-packages/smartthings-translation-agent`)
- **Self mode (크레딧 0)**: `prompt_preview.py` 가 앱 `PromptBuilder` 를 래핑해 앱과 동일한 번역/검수 프롬프트를 조립 → 에이전트가 LLM API 호출 없이 직접 번역/검수.
- **신규 래퍼 스크립트**: `glossary_manage.py`(GlossaryStore CRUD·CSV import/export), `text_workbook_create.py`(source 워크북 생성), `workbook_translate.py`·`workbook_audit.py`(앱 번역/검수 파이프라인, `--pipeline` 가드).
- **슬래시 명령어 13종(`/st-*`)**: 패키지 `commands/` 와 `.claude/commands/` 양쪽 제공. `/st-help` 로 프로젝트 개요·검수 포인트·기능 안내.
- **공용 헬퍼 `_app_pipeline.py`**: 부트스트랩·venv 재실행·시트 매핑·source group·이벤트 요약을 모아 highlight/translate/audit 가 공유.

### Changed
- `workbook_highlight_glossary.py` 를 `_app_pipeline.py` 기반으로 리팩터하고 JSON 모드 stdout 캡처·하이라이트 리포트 저장 보강.
- `SKILL.md` 에 슬래시 명령표와 셀프/파이프라인 모드 안내 추가. `references/`(rag-workflow, excel-workflow, portability, install-notes) 갱신, `self-vs-pipeline.md` 신규.

### Safety
- 용어집 쓰기(`add`/`update`/`delete`/`import`)는 명시적 `--apply` 가드 없이는 거부(`delete`·`import --mode replace` 는 비가역).
- 리포트 파일 쓰기를 `.tmp` → `os.replace()` atomic write 로 통일.
- `event_summary()` 가 `complete` 이벤트 없이 끝난 실행을 `ok` 대신 `incomplete` 로 표시.

### Verified
- 전 스크립트 `py_compile` 통과
- 크레딧 0 스모크 테스트: `prompt_preview`(번역/검수), `glossary_manage status/list`, `text_workbook_create`
- 하이라이트 회귀 테스트(산출물·원본 불변), 파이프라인 `--pipeline` 거부, 용어집 `--apply` 가드(add→delete 왕복으로 baseline 복구)

---

## [1.6.0] - 2026-06-02

### Added
- **Multi-source Integrated Pipeline**: `source_groups` 기반으로 여러 원문 시트와 타겟 시트 그룹을 한 번에 번역 및 검수할 수 있도록 지원.
- **Multi-source Highlight Pipeline**: 하이라이트 전용 모드에서도 원문 시트 그룹별 용어집 하이라이트 및 리포트 생성을 지원.

### Changed
- **Pipeline Entry Consistency**: 검수, 통합 번역+검수, 하이라이트 전용 파이프라인의 멀티소스 진입 조건을 모두 `if source_groups:`로 통일.
- **Task Payload Propagation**: 통합 번역 및 하이라이트 작업에서 FastAPI 태스크 파라미터의 `source_groups`를 `TranslationChecker`로 전달.

### Fixed
- **Highlight Multi-source Crash**: 하이라이트 전용 다중 소스 경로에서 함수 내부 `import asyncio`로 인해 발생하던 `UnboundLocalError`를 수정.

### Verified
- `venv/bin/python -m compileall -q src tests`
- `PYTHONPATH=src venv/bin/python -m unittest tests/test_prompt_builder.py`
- 다중 소스 통합 번역 스모크 테스트
- 다중 소스 하이라이트 스모크 테스트

---

## [1.4.0] - 2026-05-14

### Added
- **Prompt Module Universe**: `universe.html` 및 `/api/prompt_universe` 엔드포인트 추가. 전체 프롬프트 모듈 구조를 그래프로 시각화.
- **RAG Viewer**: `rag_viewer.html` 및 관련 엔드포인트 추가. RAG DB 내용을 브라우저에서 필터링 및 검색 가능.
- **API Key Session Storage**: `.env` 파일 없이도 브라우저 세션에 API 키를 임시 저장하여 사용할 수 있는 UI 패널 추가.
- **Japanese Navigation Path Rule**: 일본어 타겟 시 검토 시 내비게이션 경로에 `「 」`를 사용하도록 자동화 규칙 추가.

### Changed
- **Prompt Module Refactoring**: `src/translation_web_app/prompt_modules.py`의 구조를 평탄화하고 상수화하여 관리 효율성 증대.
- **Context-aware Bracket Logic**: `Title/Button` 행에 대해서는 용어집 브래킷(`[]`)을 자동으로 제외하도록 로직 고도화.
- **BX Style Enhancements**: 페르소나 및 보이스 속성(OPEN/BOLD/AUTHENTIC)의 구체적인 가이드라인 및 Negative Constraints 강화.
- **UI Layout Optimization**: 메인 페이지와 인스펙터의 레이아웃을 개선하여 가독성 및 사용성 향상.

### Fixed
- **Bracket/Glossary Bug**: 특정 조건에서 용어집 매칭 시 브래킷이 누락되거나 잘못 적용되는 문제 해결.
- **US English Period Placement**: 미국 영어 검수 시 마침표가 따옴표 밖으로 나가지 않던 규칙 중복/오류 수정.
- **Navigation Path Punctuation**: 언어별(US, Intl, JA) 마침표 위치 로직을 컨텍스트에 따라 명확히 분리.

---

## [1.3.4] - 2026-04-15
### Added
- **General Chat Prompts**: Created a comprehensive master prompt collection (`docs/prompts_for_chat.md`) allowing users to execute the translation inspection logic manually in ChatGPT or Gemini.
- **Persistent Version History**: Integrated a "Recent Updates" summary into the README while maintaining the full historical log in the docs.

### Fixed
- **Critical Syntax Error**: Resolved a `SyntaxError` in `src/translation_web_app/checker_service.py` (line 1665) caused by a bracket mismatch (`]`) within the keyword filtering logic that prevented the app from starting.

### Improved
- **Prompt Engine Documentation**: Re-organized and clarified the modular prompt architecture documentation for better developer onboarding.

---

## [1.3.3] - 2026-04-06
### Added
- **Korean RAG Auto-Detection**: Implemented automatic language detection for RAG similarity searches. If the query contains Korean characters, it defaults to the Korean source collection (`COLLECTION_KR`).
- **Global RAG Search**: Added an "All" option to the RAG Knowledge Base Viewer, allowing semantic searches across all languages simultaneously without a mandatory target filter.

### Fixed
- **Glossary Detection Logic**: Fixed a critical bug where empty header cells in the glossary CSV were incorrectly matched as the source language column (Python's `"" in "any_string"` issue).
- **Korean Glossary Matching**: Implemented dual-key registration for Korean source text. Glossary entries now map both the English key and the Korean term to the target translation, enabling correct matching for Korean source files.
- **Skip Logic Enhancement**: Updated the glossary mismatch skip logic to properly handle "x" (lowercase) in the rule/remark column, ensuring consistent behavior for deactivated terms.

### Changed
- **RAG Viewer UI**: Updated the similarity search tab to support optional target language selection and improved input validation.

---

## [1.3.2] - 2026-04-03
### Added
- **Hybrid RAG Logic**: Implemented a 2-stage retrieval process (Identity Match -> Semantic Similarity) with a user-configurable toggle to bypass 100% matches.
- **Modular Prompt Architecture**: Redesigned the prompt engine into 7 functional modules: Persona, BX Guidelines, Language Hints, RAG Context, Glossary Rules, Context-Aware Branching, and Format Constraints.
- **Architectural Documentation**: Created a new [Prompt Architecture Chart](docs/prompt_architecture.html) with a card-based visual diagram matching the GEM prompt style.
- **Model Support**: Added support for Gemini 3.1/3.0 series and GPT-5.4 models.

### Changed
- **UI Layout Redesign**: Re-organized the main dashboard into a 3-column layout:
  - **Left (Pre-settings)**: Operational modes and model selections.
  - **Center (Files & Execution)**: Main workspace including upload, glossary, sheet mapping, cell range, and RAG status.
  - **Right (Live Progress)**: Terminal and progress monitoring.
- **RAG Dashboard**: Refactored the RAG Knowledge Base section into a unified card in the center panel for better visibility.
- **Header Optimization**: Tightened the header layout for a more consolidated "app-like" feel.

### Fixed
- **Audit Model Consistency**: Fixed an issue where the reasoning model selection wasn't correctly propagated to the background audit task.
- **RAG DB Sync**: Resolved a potential sync issue when updating specific story data in the vector database.

---

## [1.2.5] - 2026-03-30
### Added
- **TXT-to-HTML Viewer Integration**: Integrated a standalone HTML visualizer into `static/viewer/` to render translation reports with rich UI.
- **AI & RAG UI Enhancements**: Implemented card-based layouts for AI evaluations and progress bars for RAG similarity visualization.
- **Dual-Format Reporting**: Updated `src/translation_web_app/checker_service.py` to output both human-readable text and hidden JSON payloads (`[상세 - AI Payload]`, `[상세 - RAG Payload]`) for the viewer to parse.
- **Version Tracking**: Added a version and last updated date label to the main UI and viewer sidebar.

### Changed
- **Branding Generalization**: Renamed all "Gemini" specific labels to model-neutral "AI" (e.g., `Gemini 검수 결과` -> `AI 검수 결과`).
- **Glossary Loader Optimization**: 
  - Added a language alias system (e.g., `Korean` <-> `ko_KR`) to handle diverse column headers.
  - Implemented regex-based high-performance screening to only inject relevant glossary terms into LLM prompts.
- **Pipeline Cleanup**: Unified `Translate+Inspect` and `Inspect-Only` modes for better consistency.

### Fixed
- **Regex Lookahead Bug**: Fixed a parsing error in `app.js` where JSON brackets (`[...]`) were incorrectly treated as new section headers, causing missing data.
- **Glossary Matching**: Resolved issues where glossaries weren't loading due to language name mismatches.

---

## [1.1.0] - Previous
- Initial RAG DB integration.
- Multi-sheet processing support.
- Gemini 2.0/2.5 API integration.
