import sys
import sqlite3
from pathlib import Path
import pytest

# Add the scripts folder to sys.path
agent_scripts_dir = Path(__file__).parent.parent / "agent-packages" / "smartthings-translation-agent" / "scripts"
sys.path.append(str(agent_scripts_dir))

import rag_lookup


@pytest.fixture
def test_db_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
    CREATE TABLE rag_pairs (
        source_text TEXT,
        target_text TEXT,
        target_lang TEXT,
        section_code TEXT,
        story_id TEXT,
        source_group TEXT,
        tone_flag TEXT
    )
    """)
    # Add target-side translations (US source_group)
    conn.execute(
        "INSERT INTO rag_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Save energy", "Energie sparen", "DE(독일)", "//sec1", "story_001", "us", None)
    )
    # Add source-side translations (KR source_group)
    conn.execute(
        "INSERT INTO rag_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("스마트홈 시작", "Start smart home", "US(미국)", "//sec2", "story_002", "kr", None)
    )
    conn.commit()
    yield conn
    conn.close()


def test_resolve_lang_code():
    assert rag_lookup.resolve_lang_code("JA(일본)") == "JA"
    assert rag_lookup.resolve_lang_code("Japanese") == "JA"
    assert rag_lookup.resolve_lang_code("DE") == "DE"
    assert rag_lookup.resolve_lang_code("KR") == "KR"


def test_offline_query_excludes_tone_flagged_by_default(test_db_conn):
    test_db_conn.execute(
        "INSERT INTO rag_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("别让例句", "别让任何事情打断你", "CN(중국)", "//sec3", "story_014", "kr", "colloquial_cn"),
    )
    test_db_conn.commit()

    excluded = rag_lookup.offline_query(
        test_db_conn, "别让例句", ["CN(중국)"], "kr", n=3
    )
    assert excluded == []

    included = rag_lookup.offline_query(
        test_db_conn, "别让例句", ["CN(중국)"], "kr", n=3, include_tone_flagged=True
    )
    assert len(included) == 1
    assert included[0]["target"] == "别让任何事情打断你"


def test_offline_query(test_db_conn):
    # Test exact match
    results = rag_lookup.offline_query(
        test_db_conn, "Save energy", ["DE(독일)"], "us", n=3
    )
    assert len(results) == 1
    assert results[0]["target"] == "Energie sparen"
    assert results[0]["match_type"] == "exact"


def test_offline_korean_source_query(test_db_conn):
    # Test KR source-side query (exact)
    results = rag_lookup.offline_korean_source_query(
        test_db_conn, "스마트홈 시작", n=3, keyword=False
    )
    assert len(results) == 1
    assert results[0]["korean_text"] == "스마트홈 시작"
    assert results[0]["paired_target"] == "Start smart home"
    assert results[0]["lookup_side"] == "source"

    # Test KR source-side query (keyword)
    results_kw = rag_lookup.offline_korean_source_query(
        test_db_conn, "스마트홈", n=3, keyword=True
    )
    assert len(results_kw) == 1
    assert results_kw[0]["korean_text"] == "스마트홈 시작"
    assert results_kw[0]["match_type"] == "source_keyword"


def test_lookup_router_with_korean(monkeypatch, test_db_conn):
    # Mock database connection
    monkeypatch.setattr(rag_lookup, "_connect_sqlite", lambda: test_db_conn)

    # Build argparse namespace like arguments
    class Args:
        query = "스마트홈"
        target_lang = "KR"
        source_lang = "Korean"
        n = 3
        mode = "auto"
        keyword = True
        story = None
        section = None
        app_root = None
        json = True
        include_tone_flagged = False

    args = Args()
    res = rag_lookup.lookup(args)
    
    assert res["lookup_side"] == "source"
    assert len(res["examples"]) == 1
    assert res["examples"][0]["korean_text"] == "스마트홈 시작"
    assert res["examples"][0]["paired_target"] == "Start smart home"
