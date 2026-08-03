#!/usr/bin/env python3
"""Stage, search, publish, and track SmartThings review reports in Obsidian.

The report Markdown and its result manifest remain the source of truth. A Base
is a view over report front matter, never an approval store. All vault writes
require an explicit ``--apply`` flag.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UPSTREAM_REPO = "https://github.com/kepano/obsidian-skills"
REQUIRED_SKILLS = ("obsidian-markdown", "obsidian-cli", "obsidian-bases")
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read(path: str | Path) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_read(path))
    if not isinstance(payload, dict):
        raise ValueError("result manifest는 JSON object여야 합니다.")
    return payload


def _frontmatter(text: str) -> tuple[str, str]:
    match = _FRONTMATTER.match(text)
    return (match.group(0), text[match.end():]) if match else ("", text)


def _with_properties(text: str) -> str:
    """Add only missing Obsidian properties; retain the report contract verbatim."""
    front, body = _frontmatter(text)
    if not front:
        front = "---\nreport_schema_version: 1\nworkflow: agent_review\nstatus: draft\n---\n"
    additions: list[str] = []
    if not re.search(r"^tags:\s*$", front, re.MULTILINE):
        additions.extend(["tags:", "  - smartthings", "  - localization", "  - translation-review", "  - obsidian-report"])
    if not re.search(r"^obsidian_workflow:\s*", front, re.MULTILINE):
        additions.append("obsidian_workflow: staged")
    if additions:
        front = front[:-4].rstrip() + "\n" + "\n".join(additions) + "\n---\n"
    if "## Agent Notes" not in body:
        body = body.rstrip() + "\n\n## Agent Notes\n\n> [!note] Obsidian 작업 메모\n> 이 노트는 표준 검수 리포트와 승인 manifest를 보조합니다. 규칙·glossary·승인 상태의 단일 기준은 구조화 finding과 manifest입니다.\n"
    if "## Decision Log" not in body:
        body = body.rstrip() + "\n\n## Decision Log\n\n- " + datetime.now(timezone.utc).isoformat() + ": Obsidian 초안 생성 (vault 미발행)\n"
    return front + body.lstrip("\n")


def stage_report(report: str | Path, output: str | Path) -> dict[str, Any]:
    source = Path(report).expanduser()
    if not source.is_file():
        raise FileNotFoundError("리포트 파일을 찾을 수 없습니다.")
    destination = Path(output).expanduser()
    _atomic_write(destination, _with_properties(_read(source)))
    return {"status": "ok", "mode": "stage", "source": str(source), "draft": str(destination)}


def _safe_vault_destination(vault: Path, relative: str, suffix: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix.lower() != suffix:
        raise ValueError(f"vault 대상은 vault 내부의 상대 {suffix} 경로여야 합니다.")
    resolved_vault = vault.resolve()
    destination = (resolved_vault / candidate).resolve()
    if resolved_vault not in destination.parents:
        raise ValueError("vault 밖 경로에는 쓸 수 없습니다.")
    return destination


def _locale_block(text: str, locale: str) -> str | None:
    header = re.compile(rf"^##\s+{re.escape(locale)}(?:\b|\s|\()", re.MULTILINE)
    match = header.search(text)
    if not match:
        return None
    next_heading = re.compile(r"^##\s+", re.MULTILINE).search(text, match.end())
    return text[match.start(): next_heading.start() if next_heading else len(text)].rstrip() + "\n"


def _replace_locale_block(existing: str, incoming: str, locale: str) -> str:
    incoming_block = _locale_block(incoming, locale)
    if not incoming_block:
        raise ValueError(f"초안에 {locale} 언어별 섹션이 없습니다.")
    match = re.compile(rf"^##\s+{re.escape(locale)}(?:\b|\s|\()", re.MULTILINE).search(existing)
    if not match:
        return existing.rstrip() + "\n\n" + incoming_block
    next_heading = re.compile(r"^##\s+", re.MULTILINE).search(existing, match.end())
    end = next_heading.start() if next_heading else len(existing)
    return existing[:match.start()] + incoming_block + "\n" + existing[end:].lstrip("\n")


def publish_report(draft: str | Path, vault: str | Path, destination: str, *, apply: bool, locale: str | None = None) -> dict[str, Any]:
    if not apply:
        raise PermissionError("vault 발행은 --apply와 사용자 명시 승인이 필요합니다.")
    vault_path = Path(vault).expanduser()
    source = Path(draft).expanduser()
    if not vault_path.is_dir() or not source.is_file():
        raise FileNotFoundError("Obsidian vault 또는 발행할 초안을 찾을 수 없습니다.")
    target = _safe_vault_destination(vault_path, destination, ".md")
    rendered, operation = _read(source), "created"
    if target.exists():
        if not locale:
            raise ValueError("기존 리포트 갱신에는 --locale이 필요합니다.")
        rendered, operation = _replace_locale_block(_read(target), rendered, locale), "locale_updated"
    _atomic_write(target, rendered)
    return {"status": "ok", "mode": "publish", "operation": operation, "report": str(target), "locale": locale}


def _filesystem_search(vault: Path, query: str, limit: int) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    needle = query.casefold()
    for note in sorted(vault.rglob("*.md")):
        try:
            for number, line in enumerate(note.read_text(encoding="utf-8").splitlines(), start=1):
                if needle in line.casefold():
                    hits.append({"path": str(note.relative_to(vault)), "line": number, "text": line.strip()})
                    if len(hits) >= limit:
                        return hits
        except UnicodeDecodeError:
            continue
    return hits


def search_vault(vault: str | Path, query: str, *, limit: int, vault_name: str | None = None) -> dict[str, Any]:
    vault_path = Path(vault).expanduser()
    if not vault_path.is_dir() or not query.strip():
        raise ValueError("검색할 vault 경로와 검색어를 확인하세요.")
    cli = shutil.which("obsidian")
    if cli and vault_name:
        completed = subprocess.run([cli, f"vault={vault_name}", "search", f"query={query}", f"limit={limit}"], capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            return {"status": "ok", "mode": "obsidian_cli", "query": query, "results": completed.stdout.strip()}
    return {"status": "ok", "mode": "filesystem_fallback", "query": query, "results": _filesystem_search(vault_path, query, limit)}


def _delivery_is_valid(payload: dict[str, Any]) -> bool:
    return bool(payload.get("status") == "ok" and payload.get("final") and payload.get("glossary") and (payload.get("delivery_manifest") or payload.get("result_manifest") or payload.get("highlight_report")))


def sync_status(report: str | Path, result_manifest: str | Path, output: str | Path) -> dict[str, Any]:
    payload = _load_json(result_manifest)
    if not _delivery_is_valid(payload):
        raise ValueError("유효한 delivery/review-apply result manifest가 아닙니다.")
    front, body = _frontmatter(_read(report))
    if not front:
        raise ValueError("표준 리포트 front matter가 필요합니다.")
    front = re.sub(r"^status:\s*.*$", "status: applied", front, flags=re.MULTILINE)
    entry = (
        f"- {datetime.now(timezone.utc).isoformat()}: delivery 적용 검증 완료\n"
        f"  - result_manifest: `{Path(result_manifest).name}`\n"
        f"  - final: `{Path(str(payload['final'])).name}`\n"
        f"  - glossary: `{Path(str(payload['glossary'])).name}`\n"
        f"  - highlight_report: `{Path(str(payload.get('highlight_report') or '')).name}`\n"
    )
    if "## Decision Log" not in body:
        body = body.rstrip() + "\n\n## Decision Log\n"
    destination = Path(output).expanduser()
    _atomic_write(destination, front + body.rstrip() + "\n" + entry)
    return {"status": "ok", "mode": "sync_status", "report": str(destination), "result_manifest": str(result_manifest)}


def init_base(vault: str | Path, relative: str, *, apply: bool) -> dict[str, Any]:
    if not apply:
        raise PermissionError("Base 생성은 --apply와 사용자 명시 승인이 필요합니다.")
    vault_path = Path(vault).expanduser()
    if not vault_path.is_dir():
        raise FileNotFoundError("Obsidian vault 경로를 찾을 수 없습니다.")
    target = _safe_vault_destination(vault_path, relative, ".base")
    if target.exists():
        return {"status": "ok", "mode": "init_base", "operation": "exists", "base": str(target)}
    base = """filters:
  and:
    - 'file.ext == "md"'
    - 'project == "SmartThings Translation"'
    - 'file.hasTag("translation-review")'
properties:
  report_id:
    displayName: "Report"
  story_id:
    displayName: "Story"
  target_locales:
    displayName: "Locales"
  status:
    displayName: "Status"
  generated_at:
    displayName: "Generated"
views:
  - type: table
    name: "All reviews"
    order: [file.name, report_id, story_id, target_locales, status, generated_at]
  - type: table
    name: "Approval queue"
    filters:
      or: ['status == "draft"', 'status == "reviewed"', 'status == "approved"']
    order: [file.name, status, target_locales, generated_at]
  - type: table
    name: "Applied"
    filters: 'status == "applied"'
    order: [file.name, target_locales, generated_at]
"""
    _atomic_write(target, base)
    return {"status": "ok", "mode": "init_base", "operation": "created", "base": str(target)}


def status() -> dict[str, Any]:
    root = Path.home() / ".codex" / "skills"
    return {"status": "ok", "required_skills": {name: (root / name / "SKILL.md").is_file() for name in REQUIRED_SKILLS}, "obsidian_cli": shutil.which("obsidian"), "upstream": UPSTREAM_REPO, "fallback": "workspace draft and filesystem search"}


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartThings Obsidian report workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    stage = sub.add_parser("stage"); stage.add_argument("report"); stage.add_argument("--output", required=True)
    search = sub.add_parser("search"); search.add_argument("vault"); search.add_argument("query"); search.add_argument("--limit", type=int, default=20); search.add_argument("--vault-name")
    publish = sub.add_parser("publish"); publish.add_argument("draft"); publish.add_argument("vault"); publish.add_argument("destination"); publish.add_argument("--locale"); publish.add_argument("--apply", action="store_true")
    sync = sub.add_parser("sync-status"); sync.add_argument("report"); sync.add_argument("result_manifest"); sync.add_argument("--output", required=True)
    base = sub.add_parser("init-base"); base.add_argument("vault"); base.add_argument("--output", default="SmartThings Translation Reviews.base"); base.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "status": result = status()
        elif args.command == "stage": result = stage_report(args.report, args.output)
        elif args.command == "search": result = search_vault(args.vault, args.query, limit=args.limit, vault_name=args.vault_name)
        elif args.command == "publish": result = publish_report(args.draft, args.vault, args.destination, apply=args.apply, locale=args.locale)
        elif args.command == "sync-status": result = sync_status(args.report, args.result_manifest, args.output)
        else: result = init_base(args.vault, args.output, apply=args.apply)
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
