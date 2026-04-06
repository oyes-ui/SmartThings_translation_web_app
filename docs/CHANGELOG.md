# Changelog - SmartThings Translation Checker

## [v1.3.3] - 2026-04-06
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

## [v1.3.2] - 2026-04-03
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

## [v1.2.5] - 2026-03-30
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

## [v1.1.0] - Previous
- Initial RAG DB integration.
- Multi-sheet processing support.
- Gemini 2.0/2.5 API integration.
