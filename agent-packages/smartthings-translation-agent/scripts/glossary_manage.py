#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glossary_manage.py — 용어집 CRUD / 조회 / CSV import·export (크레딧 0)

앱의 GlossaryStore(services/glossary_store.py)를 그대로 호출하는 얇은 래퍼다.
용어집 로직은 재구현하지 않는다. LLM 호출 없음.

서브커맨드 (읽기는 즉시, 쓰기는 --apply 필수):
  status                      DB 현황(용어/로케일 수, 경로)            [읽기]
  locales                     로케일(언어 컬럼) 목록                  [읽기]
  list   [--search S] [--limit N] [--offset K]   용어 목록/검색       [읽기]
  add    --source-key K [--rule R] [--translations JSON] --apply       [쓰기]
  update --id ID --source-key K [--rule R] [--translations JSON] --apply [쓰기]
  delete --id ID --apply                                               [쓰기]
  import --csv PATH [--mode merge|replace] --apply                     [쓰기]
  export --out PATH                                                    [읽기]

안전: add/update/delete/import 는 --apply 없이는 거부된다(변경 미리보기만). delete 와
import --mode replace 는 비가역이므로 반드시 사용자 승인 후 --apply 를 붙인다.

사용 예:
  python scripts/glossary_manage.py status --json
  python scripts/glossary_manage.py list --search AI --json
  python scripts/glossary_manage.py add --source-key "Matter" --rule "no bracket" \
      --translations '{"3":"Matter"}' --apply
  python scripts/glossary_manage.py export --out /tmp/glossary_dump.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _app_pipeline as ap


def _store():
    from translation_web_app.services.glossary_store import GlossaryStore

    return GlossaryStore()


def _parse_translations(raw: str | None) -> dict:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--translations 는 {locale_id: value} JSON 객체여야 합니다.")
    return {str(k): str(v) for k, v in data.items()}


WRITE_COMMANDS = {"add", "update", "delete", "import"}


def _refusal_message(cmd: str, args) -> str:
    if cmd == "add":
        what = f"용어 추가: source_key='{args.source_key}'"
    elif cmd == "update":
        what = f"용어 수정: id={args.id}, source_key='{args.source_key}'"
    elif cmd == "delete":
        what = f"용어 삭제: id={args.id} (되돌리기 어려움)"
    elif cmd == "import":
        extra = " — mode=replace 는 기존 용어집을 전부 덮어씀(되돌리기 어려움)" if args.mode == "replace" else ""
        what = f"CSV import: {args.csv} (mode={args.mode}){extra}"
    else:
        what = cmd
    return (
        f"이 작업은 용어집 DB를 변경합니다 → {what}\n"
        "변경 내역을 사용자에게 보여주고 승인받은 뒤 --apply 를 붙여 다시 실행하세요."
    )


def run(args) -> dict:
    app_root = ap.bootstrap_project(args.app_root)
    ap.maybe_reexec_with_app_venv(app_root)

    cmd = args.command
    # 쓰기 작업은 명시적 --apply 가드 없이는 거부 (delete / import --mode replace 는 비가역)
    if cmd in WRITE_COMMANDS and not getattr(args, "apply", False):
        return {
            "status": "refused",
            "command": cmd,
            "reason": "apply_flag_required",
            "message": _refusal_message(cmd, args),
        }

    from dotenv import load_dotenv

    load_dotenv(app_root / ".env")
    store = _store()
    if cmd == "status":
        return {"status": "ok", "command": cmd, "result": store.status()}
    if cmd == "locales":
        return {"status": "ok", "command": cmd, "result": store.list_locales()}
    if cmd == "list":
        return {
            "status": "ok",
            "command": cmd,
            "result": store.list_terms(search=args.search or "", limit=args.limit, offset=args.offset),
        }
    if cmd == "add":
        payload = {
            "source_key": args.source_key,
            "rule_text": args.rule or "",
            "translations": _parse_translations(args.translations),
        }
        return {"status": "ok", "command": cmd, "result": store.create_term(payload)}
    if cmd == "update":
        payload = {
            "source_key": args.source_key,
            "rule_text": args.rule or "",
            "translations": _parse_translations(args.translations),
        }
        return {"status": "ok", "command": cmd, "result": store.update_term(int(args.id), payload)}
    if cmd == "delete":
        store.delete_term(int(args.id))
        return {"status": "ok", "command": cmd, "result": {"deleted": int(args.id)}}
    if cmd == "import":
        csv_path = Path(args.csv).expanduser()
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV를 찾을 수 없습니다: {csv_path}")
        return {"status": "ok", "command": cmd, "result": store.import_csv(csv_path, mode=args.mode)}
    if cmd == "export":
        out = store.export_csv(Path(args.out).expanduser())
        return {"status": "ok", "command": cmd, "result": {"export_path": str(out)}}

    raise ValueError(f"알 수 없는 서브커맨드: {cmd}")


def main() -> None:
    # 공통 옵션을 부모 파서에 두어 서브커맨드 앞/뒤 어디서든 --json/--app-root 가 먹히게 한다.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--app-root", help="app repo 경로 명시")
    common.add_argument("--json", action="store_true", help="JSON 출력")

    parser = argparse.ArgumentParser(
        description="SmartThings 용어집 관리 (앱 GlossaryStore 래퍼, 크레딧 0)", parents=[common]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="DB 현황", parents=[common])
    sub.add_parser("locales", help="로케일 목록", parents=[common])

    p_list = sub.add_parser("list", help="용어 목록/검색", parents=[common])
    p_list.add_argument("--search", help="검색어(source_key/rule LIKE)")
    p_list.add_argument("--limit", type=int, default=200)
    p_list.add_argument("--offset", type=int, default=0)

    apply_help = "DB 쓰기 확인 플래그 (없으면 거부, 승인 후 추가)"

    p_add = sub.add_parser("add", help="용어 추가", parents=[common])
    p_add.add_argument("--source-key", required=True)
    p_add.add_argument("--rule")
    p_add.add_argument("--translations", help='{"locale_id": "value"} JSON')
    p_add.add_argument("--apply", action="store_true", help=apply_help)

    p_upd = sub.add_parser("update", help="용어 수정", parents=[common])
    p_upd.add_argument("--id", required=True)
    p_upd.add_argument("--source-key", required=True)
    p_upd.add_argument("--rule")
    p_upd.add_argument("--translations", help='{"locale_id": "value"} JSON')
    p_upd.add_argument("--apply", action="store_true", help=apply_help)

    p_del = sub.add_parser("delete", help="용어 삭제", parents=[common])
    p_del.add_argument("--id", required=True)
    p_del.add_argument("--apply", action="store_true", help=apply_help)

    p_imp = sub.add_parser("import", help="CSV import", parents=[common])
    p_imp.add_argument("--csv", required=True)
    p_imp.add_argument("--mode", choices=["merge", "replace"], default="merge")
    p_imp.add_argument("--apply", action="store_true", help=apply_help)

    p_exp = sub.add_parser("export", help="CSV export", parents=[common])
    p_exp.add_argument("--out", required=True)

    args = parser.parse_args()

    # 쓰기 작업은 사용자 승인 전제 (에이전트가 SKILL 안전규칙에 따라 확인)
    try:
        res = run(args)
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2))
        else:
            print(f"❌ {e}")
        sys.exit(1)

    if res.get("status") == "refused":
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            print(f"⛔ {res['message']}")
        sys.exit(2)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(f"✅ {res['command']}")
        print(json.dumps(res["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
