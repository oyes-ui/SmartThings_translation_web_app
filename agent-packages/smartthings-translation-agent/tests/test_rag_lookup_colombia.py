from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rag_lookup  # noqa: E402


def _args(target_lang: str) -> SimpleNamespace:
    return SimpleNamespace(
        query=None,
        target_lang=target_lang,
        source_lang="English",
        n=3,
        mode="offline",
        keyword=False,
        story="story_001",
        section=None,
        include_tone_flagged=False,
    )


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE rag_pairs (
            source_text TEXT, target_text TEXT, target_lang TEXT,
            section_code TEXT, story_id TEXT, source_group TEXT,
            tone_flag TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO rag_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Save energy", "Save energy", "UK(영국)", "//sec", "story_001", "us", None),
    )
    conn.commit()
    return conn


def test_es_co_locale_alias_resolves_to_co() -> None:
    assert rag_lookup.resolve_lang_code("es_CO") == "CO"
    assert rag_lookup.resolve_lang_code("es-CO") == "CO"


def test_missing_co_variant_does_not_leak_another_language(monkeypatch) -> None:
    conn = _db()
    monkeypatch.setattr(rag_lookup, "_connect_sqlite", lambda: conn)
    monkeypatch.setattr(rag_lookup, "_has_api_key", lambda: False)

    result = rag_lookup.lookup(_args("Spanish_Colombia"))

    assert result["resolved_code"] == "CO"
    assert result["db_variants_matched"] == []
    assert result["examples"] == []
    assert any("해당하는 target_lang 이 DB" in note for note in result["notes"])
    conn.close()


def test_es_co_returns_only_colombia_rows(monkeypatch) -> None:
    conn = _db()
    conn.execute(
        "INSERT INTO rag_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Save energy", "Ahorra energía", "CO(콜롬비아)", "//sec", "story_001", "us", None),
    )
    conn.commit()
    monkeypatch.setattr(rag_lookup, "_connect_sqlite", lambda: conn)
    monkeypatch.setattr(rag_lookup, "_has_api_key", lambda: False)

    result = rag_lookup.lookup(_args("es_CO"))

    assert [row["target_lang"] for row in result["examples"]] == ["CO(콜롬비아)"]
    conn.close()
