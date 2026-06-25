#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_workbook_create.py — 구조화된 텍스트로 source 워크북(.xlsx) 생성 (크레딧 0)

앱의 create_text_source_workbook(services/text_workbook_service.py)를 호출하는
얇은 래퍼다. story(title/description) + 최대 4개 section(title/description/
disclaimer/button)을 템플릿에 채워 source 시트 워크북을 만든다. LLM 호출 없음.

입력 JSON 스키마 (--spec 파일 또는 --spec-json 인라인):
  {
    "source_sheet": "US(미국)",
    "story_number": 25,
    "update_date": "2026-06-25",
    "story": {"title": "...", "description": "..."},
    "sections": [
      {"title": "...", "description": "...", "disclaimer": "...", "button": "..."},
      ... (최대 4개)
    ]
  }

사용 예:
  python scripts/text_workbook_create.py --spec story_spec.json --json
  python scripts/text_workbook_create.py --spec-json '{"source_sheet":"US(미국)","story_number":1,"story":{"title":"Hi"},"sections":[]}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _app_pipeline as ap


def _load_spec(args) -> dict:
    if args.spec:
        data = json.loads(Path(args.spec).expanduser().read_text(encoding="utf-8"))
    elif args.spec_json:
        data = json.loads(args.spec_json)
    else:
        raise ValueError("--spec <파일> 또는 --spec-json <JSON> 중 하나가 필요합니다.")
    if not isinstance(data, dict):
        raise ValueError("spec 은 JSON 객체여야 합니다.")
    return data


def run(args) -> dict:
    app_root = ap.bootstrap_project(args.app_root)
    ap.maybe_reexec_with_app_venv(app_root)

    from dotenv import load_dotenv
    from translation_web_app.services.text_workbook_service import create_text_source_workbook

    load_dotenv(app_root / ".env")
    spec = _load_spec(args)

    source_sheet = spec.get("source_sheet")
    if not source_sheet:
        raise ValueError("spec.source_sheet 가 필요합니다. 예: 'US(미국)'")

    kwargs = dict(
        source_sheet=source_sheet,
        story_number=spec.get("story_number"),
        update_date=spec.get("update_date"),
        story=spec.get("story") or {},
        sections=spec.get("sections") or [],
    )
    if args.output_dir:
        kwargs["output_dir"] = Path(args.output_dir).expanduser()

    generated = create_text_source_workbook(**kwargs)
    return {
        "status": "ok",
        "credits": 0,
        "path": str(generated.path),
        "file_id": generated.file_id,
        "file_name": generated.file_name,
        "story_id": generated.story_id,
        "source_sheet": source_sheet,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="구조화 텍스트 → source 워크북 생성 (크레딧 0)")
    parser.add_argument("--spec", help="입력 JSON 파일 경로")
    parser.add_argument("--spec-json", help="입력 JSON 인라인 문자열")
    parser.add_argument("--output-dir", help="출력 디렉터리(기본: 앱 runtime/uploads)")
    parser.add_argument("--app-root", help="app repo 경로 명시")
    parser.add_argument("--json", action="store_true", help="JSON 출력")
    args = parser.parse_args()

    try:
        res = run(args)
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2))
        else:
            print(f"❌ {e}")
        sys.exit(1)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print("✅ source 워크북 생성 완료")
        print(f"   파일: {res['path']}")
        print(f"   story_id: {res['story_id']} / source: {res['source_sheet']}")


if __name__ == "__main__":
    main()
