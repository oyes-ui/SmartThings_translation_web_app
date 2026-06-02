"""Shared filesystem paths for the SmartThings translation web app."""

from __future__ import annotations

import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

STATIC_DIR = APP_DIR / "static"
TEMPLATE_DIR = APP_DIR / "assets" / "templates"
SOURCE_WORKBOOK_TEMPLATE = TEMPLATE_DIR / "source_workbook_blank.xlsx"


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value:
        return Path(value).expanduser().resolve()
    return default


UPLOAD_DIR = _env_path("UPLOAD_DIR", PROJECT_ROOT / "runtime" / "uploads")
GLOSSARY_DIR = _env_path("GLOSSARY_DIR", PROJECT_ROOT / "runtime" / "glossary")
GLOSSARY_DB_PATH = _env_path("GLOSSARY_DB_PATH", GLOSSARY_DIR / "glossary_store.db")
LATEST_GLOSSARY_CSV = _env_path("LATEST_GLOSSARY_CSV", GLOSSARY_DIR / "latest_glossary.csv")

_DEFAULT_RAG_DB_DIR = PROJECT_ROOT / "runtime" / "rag_db"
LEGACY_RAG_DB_DIR = PROJECT_ROOT / "rag_db"
RAG_DB_DIR = _env_path("RAG_DB_DIR", _DEFAULT_RAG_DB_DIR)

_DEFAULT_EXCEL_DATA_DIR = PROJECT_ROOT / "data" / "excel"
_LEGACY_EXCEL_DATA_DIR = PROJECT_ROOT / "@translation_data" / "@excel"
if os.getenv("EXCEL_DATA_DIR"):
    EXCEL_DATA_DIR = Path(os.getenv("EXCEL_DATA_DIR")).expanduser().resolve()
elif _DEFAULT_EXCEL_DATA_DIR.exists():
    EXCEL_DATA_DIR = _DEFAULT_EXCEL_DATA_DIR
elif _LEGACY_EXCEL_DATA_DIR.exists():
    EXCEL_DATA_DIR = _LEGACY_EXCEL_DATA_DIR
else:
    EXCEL_DATA_DIR = _DEFAULT_EXCEL_DATA_DIR


CHROMA_DIR = RAG_DB_DIR / "chroma"
SQLITE_PATH = RAG_DB_DIR / "rag_store.db"


def ensure_runtime_dirs() -> None:
    """Create local runtime directories required by the app."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
    RAG_DB_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
