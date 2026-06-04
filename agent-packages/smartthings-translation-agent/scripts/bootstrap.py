#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bootstrap.py — SmartThings Translation Web App repo 탐색 & 환경 점검

이 skill 은 standalone 이 아니라 app repo 를 조작하는 app-aware 에이전트 인터페이스다.
bootstrap 은 app repo 를 찾고(launcher) 환경 상태를 점검(operator)해, 현재 가능한
동작 레벨(Level 1~4)을 알려준다. 규칙/RAG/Excel 로직을 복제하지 않는다.

표준 라이브러리만 사용 → 키·무거운 의존성 없이 항상 실행 가능.

app_root 결정 우선순위:
  1) --app-root CLI 인자
  2) 환경변수 SMARTTHINGS_APP_ROOT
  3) 저장된 config (~/.smartthings_translation_agent/config.json 의 app_root)
  4) 현재 cwd 및 상위 디렉터리 탐색
  5) 이 스크립트 파일 위치의 상위 탐색 (skill 이 repo 안에 설치된 경우)

동작 레벨:
  Level 1  Excel-only      : openpyxl 만 (app repo 불필요)
  Level 2  Offline RAG     : app repo + rag_store.db (API 키 불필요)
  Level 3  Semantic RAG    : app repo + venv + chroma + Gemini API key
  Level 4  Full pipeline   : Level 3 + glossary (app 실행 가능)

사용 예:
  python bootstrap.py --json
  python bootstrap.py --app-root /path/to/app --save
  python bootstrap.py            # 사람이 읽는 요약
"""

import os
import sys
import json
import argparse
from pathlib import Path


CONFIG_DIR = Path.home() / ".smartthings_translation_agent"
CONFIG_PATH = CONFIG_DIR / "config.json"

# app repo 판별 마커
APP_MARKER = Path("src") / "translation_web_app"
RULES_REL = Path("docs") / "comprehensive_rules.md"
REQUIREMENTS_REL = Path("requirements.txt")


# ─── config 저장/로드 ─────────────────────────────────────────────────────────
def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(app_root: Path) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"app_root": str(Path(app_root).resolve())}
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)  # atomic write
    return CONFIG_PATH


# ─── app_root 탐색 ────────────────────────────────────────────────────────────
def is_app_root(path: Path) -> bool:
    """핵심 마커(src/translation_web_app)가 있으면 app repo 로 간주."""
    try:
        return (Path(path) / APP_MARKER).is_dir()
    except Exception:
        return False


def _search_upward(start: Path) -> Path | None:
    start = Path(start).resolve()
    for parent in [start, *start.parents]:
        if is_app_root(parent):
            return parent
    return None


def cli_app_root_from_argv(argv=None) -> str | None:
    """argparse 이전에 --app-root 값을 미리 추출 (다른 스크립트의 부트스트랩용)."""
    argv = list(sys.argv if argv is None else argv)
    for i, a in enumerate(argv):
        if a == "--app-root" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--app-root="):
            return a.split("=", 1)[1]
    return None


def resolve_app_root(cli_app_root: str | None = None) -> tuple[Path | None, str]:
    """우선순위대로 app_root 를 해석. (경로, 출처) 반환. 못 찾으면 (None, 'not-found')."""
    # 1) CLI
    if cli_app_root:
        p = Path(cli_app_root).expanduser()
        if is_app_root(p):
            return p.resolve(), "cli"
    # 2) env
    env = os.getenv("SMARTTHINGS_APP_ROOT")
    if env:
        p = Path(env).expanduser()
        if is_app_root(p):
            return p.resolve(), "env"
    # 3) config
    cfg = load_config().get("app_root")
    if cfg:
        p = Path(cfg).expanduser()
        if is_app_root(p):
            return p.resolve(), "config"
    # 4) cwd 상위 탐색
    found = _search_upward(Path.cwd())
    if found:
        return found, "cwd-search"
    # 5) 스크립트 파일 위치 상위 탐색
    found = _search_upward(Path(__file__).resolve().parent)
    if found:
        return found, "file-search"
    return None, "not-found"


# ─── 환경 점검 ────────────────────────────────────────────────────────────────
def _venv_python(app_root: Path) -> str | None:
    for rel in ("venv/bin/python", ".venv/bin/python",
                "venv/Scripts/python.exe", ".venv/Scripts/python.exe"):
        p = app_root / rel
        if p.exists():
            return str(p)
    return None


def _env_has_api_key(app_root: Path) -> bool:
    """환경변수 또는 .env 파일에 Gemini 키가 있는지 (값은 노출하지 않음)."""
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return True
    env_file = app_root / ".env"
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                key = line.split("=", 1)[0].strip()
                if key in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and "=" in line:
                    val = line.split("=", 1)[1].strip()
                    if val:
                        return True
        except Exception:
            pass
    return False


def inspect_environment(app_root: Path | None) -> dict:
    checks: dict = {
        "src_translation_web_app": False,
        "comprehensive_rules": False,
        "requirements_txt": False,
        "venv_python": None,
        "rag_store_db": False,
        "chroma_dir": False,
        "glossary_csv": False,
        "glossary_db": False,
        "env_file": False,
        "api_key_present": False,
    }
    if not app_root:
        return checks
    ar = Path(app_root)
    rag_dir = Path(os.getenv("RAG_DB_DIR")) if os.getenv("RAG_DB_DIR") else ar / "runtime" / "rag_db"
    gloss_dir = ar / "runtime" / "glossary"

    checks["src_translation_web_app"] = (ar / APP_MARKER).is_dir()
    checks["comprehensive_rules"] = (ar / RULES_REL).is_file()
    checks["requirements_txt"] = (ar / REQUIREMENTS_REL).is_file()
    checks["venv_python"] = _venv_python(ar)
    checks["rag_store_db"] = (rag_dir / "rag_store.db").is_file()
    checks["chroma_dir"] = (rag_dir / "chroma").is_dir()
    checks["glossary_csv"] = (gloss_dir / "latest_glossary.csv").is_file()
    checks["glossary_db"] = (gloss_dir / "glossary_store.db").is_file()
    checks["env_file"] = (ar / ".env").is_file()
    checks["api_key_present"] = _env_has_api_key(ar)
    return checks


def determine_level(app_root: Path | None, checks: dict) -> tuple[int, str]:
    """현재 가능한 최고 동작 레벨."""
    # Level 1 은 항상 가능 (openpyxl 만 있으면 Excel 분석)
    if not app_root or not checks["src_translation_web_app"]:
        return 1, "Excel-only (app repo 미연결)"
    has_offline = checks["rag_store_db"]
    has_semantic = has_offline and checks["chroma_dir"] and checks["api_key_present"]
    has_full = has_semantic and (checks["glossary_csv"] or checks["glossary_db"]) \
        and bool(checks["venv_python"])
    if has_full:
        return 4, "Full pipeline (RAG + glossary + LLM)"
    if has_semantic:
        return 3, "Semantic RAG (임베딩 유사검색 가능)"
    if has_offline:
        return 2, "Offline RAG (exact/keyword/메타, 키 불필요)"
    return 1, "Excel-only (RAG DB 없음)"


def build_report(cli_app_root: str | None) -> dict:
    app_root, source = resolve_app_root(cli_app_root)
    checks = inspect_environment(app_root)
    level, level_label = determine_level(app_root, checks)
    notes = []
    if not app_root:
        notes.append("app repo 를 찾지 못했습니다. --app-root <경로> 로 지정하거나 "
                     "SMARTTHINGS_APP_ROOT 환경변수를 설정하세요. (Excel-only 기능은 계속 가능)")
    else:
        if not checks["rag_store_db"]:
            notes.append("RAG DB(rag_store.db)가 없어 offline/semantic RAG 가 제한됩니다.")
        if checks["rag_store_db"] and not checks["api_key_present"]:
            notes.append("API 키가 없어 semantic 유사검색은 불가하나 offline RAG(exact/keyword)는 가능합니다.")
        if not checks["venv_python"]:
            notes.append("venv 를 찾지 못했습니다. semantic/full 기능은 의존성 설치가 필요할 수 있습니다.")
    return {
        "app_root": str(app_root) if app_root else None,
        "app_root_source": source,
        "config_path": str(CONFIG_PATH),
        "checks": checks,
        "level": level,
        "level_label": level_label,
        "notes": notes,
    }


def _print_human(rep: dict) -> None:
    print("🚀 SmartThings Translation Agent — bootstrap")
    print(f"   app_root : {rep['app_root'] or '(미발견)'}  [{rep['app_root_source']}]")
    print(f"   level    : Level {rep['level']} — {rep['level_label']}")
    print("   checks   :")
    labels = {
        "src_translation_web_app": "src/translation_web_app",
        "comprehensive_rules": "docs/comprehensive_rules.md",
        "requirements_txt": "requirements.txt",
        "venv_python": "venv python",
        "rag_store_db": "runtime/rag_db/rag_store.db",
        "chroma_dir": "runtime/rag_db/chroma/",
        "glossary_csv": "runtime/glossary/latest_glossary.csv",
        "glossary_db": "runtime/glossary/glossary_store.db",
        "env_file": ".env",
        "api_key_present": "Gemini API key",
    }
    for k, label in labels.items():
        v = rep["checks"][k]
        mark = "✅" if v else "❌"
        extra = f" ({v})" if isinstance(v, str) else ""
        print(f"     {mark} {label}{extra}")
    for n in rep["notes"]:
        print(f"   • {n}")


def main():
    p = argparse.ArgumentParser(description="SmartThings app repo 탐색 & 환경 점검")
    p.add_argument("--app-root", default=None, help="app repo 경로를 명시적으로 지정")
    p.add_argument("--save", action="store_true",
                   help="해석된 app_root 를 ~/.smartthings_translation_agent/config.json 에 저장")
    p.add_argument("--json", action="store_true", help="JSON 출력")
    args = p.parse_args()

    rep = build_report(args.app_root)

    if args.save:
        if rep["app_root"]:
            saved = save_config(Path(rep["app_root"]))
            rep["saved_config"] = str(saved)
        else:
            rep["notes"].append("app_root 가 없어 저장하지 않았습니다.")

    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        _print_human(rep)
        if rep.get("saved_config"):
            print(f"   💾 config 저장: {rep['saved_config']}")


if __name__ == "__main__":
    main()
