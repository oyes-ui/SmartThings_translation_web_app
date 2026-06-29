"""Routes for structured text input that generates a source workbook."""

from __future__ import annotations

import asyncio
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from translation_web_app.checker_service import TranslationChecker
from translation_web_app.paths import UPLOAD_DIR
from translation_web_app.services.glossary_store import resolve_glossary_file
from translation_web_app.services.text_workbook_service import create_text_source_workbook


class StoryText(BaseModel):
    title: str | None = None
    description: str | None = None


class SectionText(BaseModel):
    title: str | None = None
    description: str | None = None
    disclaimer: str | None = None
    button: str | None = None


class TextWorkbookStartRequest(BaseModel):
    source_sheet: str = "KR(한국)"
    sheets: list[str] = Field(default_factory=list)
    sheet_langs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    story_number: str | int | None = None
    update_date: str | None = None
    story: StoryText = Field(default_factory=StoryText)
    sections: list[SectionText] = Field(default_factory=list, max_length=4)
    translation_model: str = "gemini-2.5-flash"
    audit_model: str = "gpt-5.4-mini"
    translation_thinking_budget: int | None = None
    audit_reasoning_effort: str | None = None
    max_concurrency: int = 5
    bx_style_enabled: bool = False
    task_mode: str = "integrated"
    rag_identity_match: bool = True
    glossary_file_id: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None


def create_text_workbooks_router(task_store: dict) -> APIRouter:
    router = APIRouter(prefix="/api/text-workbooks", tags=["text-workbooks"])

    @router.post("/start")
    async def start_text_workbook(req: TextWorkbookStartRequest, background_tasks: BackgroundTasks):
        if req.task_mode not in {"integrated", "translate_only"}:
            raise HTTPException(status_code=400, detail="task_mode must be 'integrated' or 'translate_only'.")

        if not req.sheets:
            raise HTTPException(status_code=400, detail="At least one target sheet is required.")

        try:
            generated = create_text_source_workbook(
                source_sheet=req.source_sheet,
                story_number=req.story_number,
                update_date=req.update_date,
                story=req.story.model_dump(),
                sections=[section.model_dump() for section in req.sections],
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        task_id = str(uuid4())
        task_store[task_id] = {
            "queue": asyncio.Queue(),
            "result_path": None,
            "txt_path": None,
            "source_path": str(generated.path),
        }

        background_tasks.add_task(
            _text_workbook_translation_task,
            task_store,
            task_id,
            req,
            generated.path,
            generated.story_id,
            generated.cell_range,
        )
        return {
            "task_id": task_id,
            "source_file_id": generated.file_id,
            "source_file_name": generated.file_name,
            "story_id": generated.story_id,
        }

    return router


async def _text_workbook_translation_task(
    task_store: dict,
    task_id: str,
    params: TextWorkbookStartRequest,
    source_path: Path,
    story_id: str,
    cell_range: str,
) -> None:
    queue = task_store[task_id]["queue"]
    glossary = None
    try:
        checker = TranslationChecker(
            model_name=params.audit_model,
            max_concurrency=params.max_concurrency,
            no_backtranslation=True,
            gemini_api_key=params.gemini_api_key,
            openai_api_key=params.openai_api_key,
            audit_reasoning_effort=params.audit_reasoning_effort,
        )

        glossary = resolve_glossary_file(params.glossary_file_id)
        glossary_path = glossary.path

        await queue.put({"type": "log", "message": f"Generated source workbook from built-in template ({story_id})."})
        await queue.put({"type": "log", "message": glossary.message})
        gen = checker.run_integrated_pipeline_generator(
            source_file_path=str(source_path),
            cell_range=cell_range,
            bx_style_on=params.bx_style_enabled,
            sheet_lang_map=params.sheet_langs,
            translation_model=params.translation_model,
            audit_model=params.audit_model,
            translation_thinking_budget=params.translation_thinking_budget,
            glossary_file_path=glossary_path,
            selected_sheets=params.sheets,
            source_sheet_name=params.source_sheet,
            skip_audit=(params.task_mode == "translate_only"),
            source_lang=params.sheet_langs.get(params.source_sheet, {}).get("lang", "Korean"),
            rag_identity_match=params.rag_identity_match,
        )

        async for event in gen:
            if event["type"] != "complete":
                await queue.put(event)
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{story_id}_text_workbook_review_{timestamp}.txt"
            output_path = UPLOAD_DIR / output_filename
            output_path.write_text(event["output_data"], encoding="utf-8")

            excel_out = event.get("excel_path")
            zip_name = f"{story_id}_text_workbook_result_{timestamp}.zip"
            zip_path = UPLOAD_DIR / zip_name

            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(output_path, output_path.name)
                if excel_out and os.path.exists(excel_out):
                    zipf.write(excel_out, os.path.basename(excel_out))
                if source_path.exists():
                    zipf.write(source_path, source_path.name)

            task_store[task_id]["result_path"] = str(zip_path)
            task_store[task_id]["txt_path"] = str(output_path)
            await queue.put({
                "type": "complete",
                "download_url": f"/api/download/{task_id}",
                "result_file_id": os.path.basename(excel_out) if excel_out else None,
                "download_file_name": zip_name,
                "is_zip": True,
            })

    except Exception as exc:
        await queue.put({"type": "error", "message": f"텍스트 워크북 작업 중 오류 발생: {str(exc)}"})
    finally:
        if glossary:
            glossary.cleanup()
