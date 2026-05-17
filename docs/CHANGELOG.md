# Changelog - SmartThings Translation Checker

## [1.4.0] - 2026-05-14

### Added
- **Prompt Module Universe**: `universe.html` 및 `/api/prompt_universe` 엔드포인트 추가. 전체 프롬프트 모듈 구조를 그래프로 시각화.
- **RAG Viewer**: `rag_viewer.html` 및 관련 엔드포인트 추가. RAG DB 내용을 브라우저에서 필터링 및 검색 가능.
- **API Key Session Storage**: `.env` 파일 없이도 브라우저 세션에 API 키를 임시 저장하여 사용할 수 있는 UI 패널 추가.
- **Japanese Navigation Path Rule**: 일본어 타겟 시 검토 시 내비게이션 경로에 `「 」`를 사용하도록 자동화 규칙 추가.

### Changed
- **Prompt Module Refactoring**: `prompt_modules.py`의 구조를 평탄화하고 상수화하여 관리 효율성 증대.
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
- **Critical Syntax Error**: Resolved a `SyntaxError` in `checker_service.py` (line 1665) caused by a bracket mismatch (`]`) within the keyword filtering logic that prevented the app from starting.

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
- **Dual-Format Reporting**: Updated `checker_service.py` to output both human-readable text and hidden JSON payloads (`[상세 - AI Payload]`, `[상세 - RAG Payload]`) for the viewer to parse.
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
