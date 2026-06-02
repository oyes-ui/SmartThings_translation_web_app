"""API routes for the built-in glossary manager."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from translation_web_app.paths import UPLOAD_DIR
from translation_web_app.services.glossary_store import GlossaryStore


def create_glossaries_router(store_factory=GlossaryStore) -> APIRouter:
    router = APIRouter(prefix="/api/glossary", tags=["glossary"])

    @router.get("/status")
    async def status():
        return store_factory().status()

    @router.get("/locales")
    async def locales():
        return {"locales": store_factory().list_locales()}

    @router.get("/terms")
    async def terms(
        search: str = "",
        limit: int = Query(200, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        return store_factory().list_terms(search=search, limit=limit, offset=offset)

    @router.post("/terms")
    async def create_term(payload: dict):
        try:
            return store_factory().create_term(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.put("/terms/{term_id}")
    async def update_term(term_id: int, payload: dict):
        try:
            return store_factory().update_term(term_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/terms/{term_id}")
    async def delete_term(term_id: int):
        try:
            store_factory().delete_term(term_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True}

    @router.post("/import")
    async def import_csv(
        file: UploadFile = File(...),
        mode: str = Query("merge", pattern="^(merge|replace)$"),
    ):
        if not file.filename:
            raise HTTPException(status_code=400, detail="CSV file is required.")
        tmp_path = UPLOAD_DIR / f"glossary_import_{uuid4()}.csv"
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded glossary CSV is empty.")
        tmp_path.write_bytes(content)
        try:
            result = store_factory().import_csv(tmp_path, mode=mode)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return result

    @router.get("/export.csv")
    async def export_csv():
        output_path = UPLOAD_DIR / f"glossary_export_{uuid4()}.csv"
        exported = store_factory().export_csv(output_path)
        return FileResponse(
            exported,
            media_type="text/csv",
            filename="glossary_export.csv",
        )

    return router
