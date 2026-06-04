#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_lookup.py — SmartThings 번역 RAG 조회 (offline / semantic 이원화)

두 가지 모드:

  offline mode  (API 키 불필요)
    - SQLite(rag_store.db)만 사용. chromadb/google-genai 로드 안 함.
    - 기능: exact match, keyword 검색(LIKE), story_id / section_code / 언어 필터.
    - 키 없는 사람도 "정확 일치 과거 번역"과 메타데이터 조회가 가능.

  semantic mode (API 키 필요: GEMINI_API_KEY)
    - 기존 translation_web_app.rag_retriever(Gemini 임베딩 + ChromaDB) 사용.
    - 진짜 유사도 검색. exact + semantic 2단계.

모드 결정 (--mode auto 기본):
    - --story / --section / --keyword 중 하나라도 있으면 → offline (메타데이터 조회)
    - 키 있고 --query 있으면 → semantic
    - 키 없으면 → offline (exact→keyword fallback)

DB의 target_lang 은 시트명("JA(일본)")이며 공백/오타 변종("FR(프랑스)" vs "FR (프랑스)")이
섞여 있다. 입력 언어를 코드(JA/FR)로 정규화한 뒤 모든 변종을 후보로 조회한다.

사용 예:
  python rag_lookup.py --query "turn on the light" --target-lang JA            # auto
  python rag_lookup.py --query "절약" --keyword --target-lang JA               # offline keyword
  python rag_lookup.py --story story_001 --target-lang JA                       # offline story
  python rag_lookup.py --section //section_001_1                               # offline section
  python rag_lookup.py --query "Save energy" --target-lang JA --mode semantic   # 강제 semantic
"""

import re
import sys
import json
import sqlite3
import argparse
from pathlib import Path


# ─── 부트스트랩 (app_root 해석 + src 추가 + .env 로딩, 모두 비치명적) ──────────
def _load_env(root: Path) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
    except Exception:
        pass


def _bootstrap_project():
    """app_root 를 해석해 src/ 를 sys.path 에 추가하고 .env 로딩.

    bootstrap.py(sibling)의 resolve_app_root 를 우선 사용(--app-root/env/config/탐색).
    offline mode 는 src 없이 SQLite 만으로도 동작해야 하므로 못 찾아도 예외를 던지지 않는다.
    """
    # 1) bootstrap.py 연동 (있으면 --app-root/env/config 까지 반영)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import bootstrap as _bs
        app_root, _src = _bs.resolve_app_root(_bs.cli_app_root_from_argv())
        if app_root:
            src_dir = Path(app_root) / "src"
            if src_dir.is_dir() and str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            _load_env(Path(app_root))
            return Path(app_root)
    except Exception:
        pass

    # 2) fallback: 파일 위치 상위 탐색 (기존 동작, backward compatibility)
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "translation_web_app").is_dir():
            src_dir = parent / "src"
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            _load_env(parent)
            return parent
        if (parent / ".env").is_file():
            _load_env(parent)
    return None


PROJECT_ROOT = _bootstrap_project()


# ─── 언어 풀워드 → 코드 매핑 ──────────────────────────────────────────────────
# prompt_modules.py 는 풀워드("Japanese"), RAG DB 는 코드("JA")를 쓴다.
FULLWORD_TO_CODE = {
    "korean": "KR", "english": "US", "english_us": "US", "english_uk": "UK",
    "english_au": "AU", "english_sg": "SG", "german": "DE", "japanese": "JA",
    "french": "FR", "french_belgium": "BE", "french_canada": "CA", "italian": "IT",
    "spanish": "ES", "dutch": "NL", "swedish": "SE", "arabic": "AE", "russian": "RU",
    "turkish": "TR", "polish": "PL", "vietnamese": "VN", "thai": "TH",
    "indonesian": "ID", "chinese": "CN", "chinese_simplified": "CN",
    "chinese_traditional": "TW", "taiwanese": "TW", "portuguese": "PT",
    "portuguese_br": "BR", "portuguese_brazil": "BR",
}


def resolve_lang_code(raw: str) -> str:
    """입력을 언어 코드(괄호 앞 대문자)로 정규화. 'Japanese'/'JA'/'JA(일본)'/'FR (프랑스)' 허용."""
    if not raw:
        raise ValueError("target-lang 이 비어 있습니다.")
    s = raw.strip()
    head = re.split(r"[(（]", s, maxsplit=1)[0].strip()
    if head and re.fullmatch(r"[A-Za-z]{2,3}", head):
        return head.upper()
    key = s.lower().replace(" ", "_").replace("-", "_")
    if key in FULLWORD_TO_CODE:
        return FULLWORD_TO_CODE[key]
    if re.fullmatch(r"[A-Za-z]{2,3}", s):
        return s.upper()
    raise ValueError(
        f"언어 '{raw}' 를 코드로 해석하지 못했습니다. "
        f"코드(JA), 시트명(JA(일본)), 또는 풀워드(Japanese) 중 하나로 입력하세요."
    )


def _target_code_of(target_lang: str) -> str:
    return re.split(r"[(（]", target_lang.strip(), maxsplit=1)[0].strip().upper()


def _source_group(source_lang: str) -> str:
    """retrieve() 와 동일 규칙으로 source_group('kr'/'us'). chromadb import 없이 판정."""
    sl = (source_lang or "").lower()
    return "kr" if ("korea" in sl or sl == "ko") else "us"


# ─── SQLite 직접 접근 (offline mode: chromadb 미로드) ─────────────────────────
def _sqlite_path() -> Path:
    """rag_store.db 경로. paths.py(가벼움) 시도 후 기본 경로 fallback."""
    try:
        from translation_web_app.paths import SQLITE_PATH  # paths 는 chromadb 미의존
        return Path(SQLITE_PATH)
    except Exception:
        root = PROJECT_ROOT or Path.cwd()
        return root / "runtime" / "rag_db" / "rag_store.db"


def _connect_sqlite() -> sqlite3.Connection:
    db = _sqlite_path()
    if not db.is_file():
        raise FileNotFoundError(
            f"RAG SQLite DB를 찾을 수 없습니다: {db}\n"
            f"RAG 데이터가 없으면 offline 조회도 불가합니다. "
            f"runtime/rag_db/ 를 전달받거나 rag_db_builder 로 빌드하세요."
        )
    return sqlite3.connect(str(db))


def find_db_variants(conn: sqlite3.Connection, code: str, source_group: str) -> list[str]:
    """코드와 일치하는 모든 변종 시트명(공백/오타 포함) 반환."""
    rows = conn.execute(
        "SELECT DISTINCT target_lang FROM rag_pairs WHERE source_group = ?",
        (source_group,),
    ).fetchall()
    return [tl for (tl,) in rows if tl and _target_code_of(tl) == code]


def _row_to_example(row, match_type: str, score: float) -> dict:
    src, tgt, tlang, sec, story = row
    return {
        "source": src, "target": tgt, "target_lang": tlang,
        "section_code": sec, "story_id": story,
        "match_type": match_type, "similarity_score": score,
    }


def offline_query(
    conn, query, variants, source_group, n, keyword=False, story=None, section=None
) -> list[dict]:
    """offline 조회: exact / keyword(LIKE) / story / section / lang 필터 조합."""
    where = ["source_group = ?"]
    params: list = [source_group]

    if variants:
        where.append("target_lang IN (%s)" % ",".join("?" * len(variants)))
        params.extend(variants)
    if story:
        where.append("story_id = ?")
        params.append(story)
    if section:
        where.append("section_code = ?")
        params.append(section)

    examples: list[dict] = []

    # query 가 있으면: exact 먼저, 부족하면 keyword(LIKE) fallback (또는 --keyword 강제)
    if query and not keyword:
        w = where + ["source_text = ?"]
        rows = conn.execute(
            f"SELECT source_text,target_text,target_lang,section_code,story_id "
            f"FROM rag_pairs WHERE {' AND '.join(w)} LIMIT ?",
            (*params, query, n),
        ).fetchall()
        examples = [_row_to_example(r, "exact", 1.0) for r in rows]

    if query and (keyword or len(examples) < n):
        like = f"%{query}%"
        w = where + ["(source_text LIKE ? OR target_text LIKE ?)"]
        rows = conn.execute(
            f"SELECT source_text,target_text,target_lang,section_code,story_id "
            f"FROM rag_pairs WHERE {' AND '.join(w)} LIMIT ?",
            (*params, like, like, n * 3),
        ).fetchall()
        seen = {(e["source"], e["target_lang"]) for e in examples}
        for r in rows:
            if len(examples) >= n:
                break
            ex = _row_to_example(r, "keyword", 0.0)
            if (ex["source"], ex["target_lang"]) not in seen:
                examples.append(ex)

    # query 없이 story/section 만으로 조회 (메타데이터 lookup)
    if not query and (story or section):
        rows = conn.execute(
            f"SELECT source_text,target_text,target_lang,section_code,story_id "
            f"FROM rag_pairs WHERE {' AND '.join(where)} LIMIT ?",
            (*params, n),
        ).fetchall()
        examples = [_row_to_example(r, "lookup", 0.0) for r in rows]

    return examples[:n]


# ─── semantic mode (키 필요, RagRetriever lazy import) ───────────────────────
def semantic_query(query, variants, source_lang, n) -> list[dict]:
    from translation_web_app.rag_retriever import get_retriever  # lazy: chromadb 여기서 로드
    retriever = get_retriever()
    if not retriever.is_available():
        return []
    seen, out = set(), []
    for variant in variants:
        for ex in retriever.retrieve(query, variant, source_lang=source_lang, n_results=n):
            key = (ex.get("source"), ex.get("target"), ex.get("target_lang"))
            if key not in seen:
                seen.add(key)
                out.append(ex)
    out.sort(key=lambda e: e.get("similarity_score", 0), reverse=True)
    return out[:n]


def _has_api_key() -> bool:
    import os
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _resolve_mode(requested: str, query, keyword, story, section) -> str:
    if requested != "auto":
        return requested
    # 메타데이터/키워드 조회는 임베딩과 무관 → offline
    if story or section or keyword:
        return "offline"
    if query and _has_api_key():
        return "semantic"
    return "offline"


def lookup(args) -> dict:
    code = resolve_lang_code(args.target_lang) if args.target_lang else None
    sgroup = _source_group(args.source_lang)
    mode = _resolve_mode(args.mode, args.query, args.keyword, args.story, args.section)

    result = {
        "mode": mode,
        "query": args.query,
        "requested_target": args.target_lang,
        "resolved_code": code,
        "source_lang": args.source_lang,
        "source_group": sgroup,
        "filters": {"story": args.story, "section": args.section, "keyword": args.keyword},
        "examples": [],
        "notes": [],
    }

    if mode == "semantic":
        if not args.query or not code:
            result["notes"].append("semantic mode 는 --query 와 --target-lang 이 필요합니다.")
            return result
        if not _has_api_key():
            result["notes"].append(
                "API 키가 없어 semantic 을 쓸 수 없습니다. --mode offline 을 사용하세요."
            )
            return result
        conn = _connect_sqlite()
        try:
            variants = find_db_variants(conn, code, sgroup)
        finally:
            conn.close()
        result["db_variants_matched"] = variants
        if not variants:
            result["notes"].append(f"코드 '{code}'에 해당하는 target_lang 이 DB에 없습니다.")
            return result
        result["examples"] = semantic_query(args.query, variants, args.source_lang, args.n)
        if not result["examples"]:
            result["notes"].append("semantic 결과 없음 (RAG 비활성 또는 유사 사례 없음).")
        return result

    # offline mode
    conn = _connect_sqlite()
    try:
        variants = find_db_variants(conn, code, sgroup) if code else []
        result["db_variants_matched"] = variants
        if code and not variants:
            result["notes"].append(
                f"코드 '{code}'에 해당하는 target_lang 이 DB(source_group={sgroup})에 없습니다."
            )
        result["examples"] = offline_query(
            conn, args.query, variants, sgroup, args.n,
            keyword=args.keyword, story=args.story, section=args.section,
        )
    finally:
        conn.close()

    if len(variants) > 1:
        result["notes"].append(
            f"⚠ DB 에 '{code}' 변종이 {len(variants)}개({variants}). 모두 조회했습니다."
        )
    if not result["examples"]:
        result["notes"].append("offline 조회 결과 없음. 쿼리/필터/언어를 확인하세요.")
    if not _has_api_key():
        result["notes"].append("(API 키 없음 → offline mode. 유사도 검색은 키 설정 후 가능.)")
    return result


def _print_human(res: dict) -> None:
    print(f"🔍 mode    : {res['mode']}")
    print(f"   query   : {res['query']}")
    print(f"   target  : {res['requested_target']} → code={res['resolved_code']}")
    f = res["filters"]
    if f["story"] or f["section"] or f["keyword"]:
        print(f"   filters : story={f['story']} section={f['section']} keyword={f['keyword']}")
    if "db_variants_matched" in res:
        print(f"   variants: {res['db_variants_matched']}")
    for note in res["notes"]:
        print(f"   note    : {note}")
    print("-" * 60)
    if not res["examples"]:
        print("(결과 없음)")
        return
    for i, ex in enumerate(res["examples"], 1):
        mt = ex.get("match_type", "?")
        sim = ex.get("similarity_score", "")
        tag = mt + (f", sim={sim}" if mt == "semantic" else "")
        print(f"[{i}] ({tag}) lang={ex.get('target_lang')} "
              f"section={ex.get('section_code')} story={ex.get('story_id')}")
        print(f"    source: {ex.get('source')}")
        print(f"    target: {ex.get('target')}")


def main():
    p = argparse.ArgumentParser(description="SmartThings RAG 조회 (offline/semantic)")
    p.add_argument("--query", help="검색할 원문 텍스트 (exact 또는 keyword)")
    p.add_argument("--target-lang", help="코드(JA)/시트명(JA(일본))/풀워드(Japanese). offline 에선 생략 가능")
    p.add_argument("--source-lang", default="English", help="소스 언어 (기본 English; 'Korean'이면 KR)")
    p.add_argument("--n", type=int, default=3, help="반환 수 (기본 3)")
    p.add_argument("--mode", choices=["auto", "offline", "semantic"], default="auto",
                   help="auto(기본): 키 있고 query면 semantic, 아니면 offline")
    p.add_argument("--keyword", action="store_true", help="offline: 부분일치(LIKE) 검색")
    p.add_argument("--story", help="offline: story_id 필터 (예: story_001)")
    p.add_argument("--section", help="offline: section_code 필터 (예: //section_001_1)")
    p.add_argument("--app-root", help="app repo 경로 명시 (미지정 시 자동 탐색)")
    p.add_argument("--json", action="store_true", help="JSON 출력")
    args = p.parse_args()

    if not (args.query or args.story or args.section):
        p.error("--query, --story, --section 중 최소 하나는 필요합니다.")

    try:
        res = lookup(args)
    except (ValueError, FileNotFoundError) as e:
        msg = {"error": str(e)}
        print(json.dumps(msg, ensure_ascii=False) if args.json else f"❌ {e}")
        sys.exit(2)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        _print_human(res)


if __name__ == "__main__":
    main()
