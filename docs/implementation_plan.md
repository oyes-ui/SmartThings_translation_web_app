# Implementation Plan - AI Parallel Translation & Cross-Inference Audit

This system integrates Gemini-2.0-Flash for high-speed parallel translation and GPT-5.2-Thinking (o1) for precision reasoning-based auditing. It uses a JSON-based mapping to process multi-sheet Excel files, preserving the original structure.

## Proposed Changes

### Core Logic

#### [NEW] [mapping_manager.py](file:///Users/df_n67/Documents/2507_SAMSUNG/SmartThings_translation_web_app/mapping_manager.py)
Encapsulates `sheet_langs.json` logic.
- **Coordinate Tracking**: Instead of just extracting text, it will map each source string to its `(Row, Column)` coordinate to ensure perfectly aligned writing in target sheets.

#### [NEW] [bx_guideline_engine.py](file:///Users/df_n67/Documents/2507_SAMSUNG/SmartThings_translation_web_app/bx_guideline_engine.py)
The **Samsung BX Guideline Engine**.
- Stores Persona, Voice Attributes, and Few-shot examples.
- **Style Validation**: Logic to verify "Double Take" and "Personification" patterns in the generated output.

#### [NEW] [model_handler.py](file:///Users/df_n67/Documents/2507_SAMSUNG/SmartThings_translation_web_app/model_handler.py)
Dynamic API client management.

#### [MODIFY] [checker_service.py](file:///Users/df_n67/Documents/2507_SAMSUNG/SmartThings_translation_web_app/checker_service.py)
- **Formatting Preservation**: Use `openpyxl`'s cell-by-cell update logic to replace only the `cell.value`, preserving original fonts, colors, and borders.
- **Parallel Pipeline**: `[Translate/Transcreate -> Audit -> Write-Back]`.

### API & UI

#### [MODIFY] [main.py](file:///Users/df_n67/Documents/2507_SAMSUNG/SmartThings_translation_web_app/main.py)
- Update `StartRequest` and orchestration logic.

---

## Verification Plan

### Automated Tests
- **Zero-Shot vs Few-Shot Comparison**: Script to check if BX-enabled output shows better adherence to the "Double Take" headline pattern than standard translation.
- **Formatting Check**: Load the output Excel and verify that styles (cell colors, font weights) match the source Excel.

### Manual Verification
- Upload a sample Excel with multiple language sheets.
- Select Gemini and GPT models.
- Verify that the resulting ZIP contains:
    1.  A single Excel file where all target sheets (other than KR) are populated.
    2.  A `.txt` report file containing GPT's reasoning for each translation.
- Check the GPT reasoning in the report to ensure it mentions Samsung's brand tone or specific nuances as requested.
