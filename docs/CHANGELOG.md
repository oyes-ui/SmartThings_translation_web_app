# Changelog - SmartThings Translation Checker

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
