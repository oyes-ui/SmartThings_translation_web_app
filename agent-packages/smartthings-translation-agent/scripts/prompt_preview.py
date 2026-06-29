#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_preview.py — 앱과 동일한 번역/검수 프롬프트를 조립해서 출력 (크레딧 0)

셀프 모드의 핵심 도구다. 앱의 PromptBuilder(prompt_builder.py)를 그대로 호출해
Gemini/GPT로 보낼 것과 **동일한** 규칙·포맷·BX·현지화 프롬프트를 텍스트로 만든다.
에이전트(Claude)는 이 프롬프트 + 용어집(glossary_manage.py) + RAG 사례(rag_lookup.py)
를 받아 LLM 크레딧 없이 직접 번역/검수를 수행한다.

이 스크립트는 LLM을 호출하지 않는다. 프롬프트 텍스트만 만든다.

사용 예:
  # 번역 프롬프트 (독일어, description 맥락)
  python scripts/prompt_preview.py --text "Turn on the light" --target-lang "DE(독일)" --row-key description

  # RAG 컨텍스트 주입 (rag_lookup.py 결과 문자열을 --rag-context 로)
  python scripts/prompt_preview.py --text "Save energy" --target-lang "JA(일본)" --rag-context "..." --json

  # 검수(audit) 프롬프트
  python scripts/prompt_preview.py --audit --text "Turn on the light" --translated "Licht einschalten" \
      --target-lang "DE(독일)" --row-key title
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _app_pipeline as ap


def build(args) -> dict:
    app_root = ap.bootstrap_project(args.app_root)
    ap.maybe_reexec_with_app_venv(app_root)

    from translation_web_app.prompt_builder import PromptBuilder

    builder = PromptBuilder()
    # glossary_context 는 프롬프트의 용어집 포맷 섹션 토글 용도(truthy 여부만 사용).
    glossary_ctx = True if args.glossary else None

    if args.audit:
        prompt = builder.build_audit_prompt(
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            target_lang_code=args.target_lang_code,
            row_key=args.row_key,
            glossary_context=glossary_ctx,
        )
        mode = "audit"
    else:
        prompt = builder.build_translation_prompt(
            target_lang=args.target_lang,
            source_lang=args.source_lang,
            bx_style_on=args.bx,
            rag_context=args.rag_context,
            row_key=args.row_key,
            glossary_context=glossary_ctx,
        )
        mode = "translation"

    ctx_mode = builder.get_glossary_context_mode(args.row_key)
    wrap = builder.should_wrap_glossary(args.row_key)
    brackets = builder.get_brackets(args.target_lang)

    return {
        "status": "ok",
        "mode": mode,
        "credits": 0,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "row_key": args.row_key,
        "glossary_context_mode": ctx_mode,
        "wrap_glossary_brackets": wrap,
        "brackets": brackets,
        "source_text": args.text,
        "translated_text": args.translated if args.audit else None,
        "prompt": prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="앱과 동일한 번역/검수 프롬프트 조립 — 에이전트 셀프 번역/검수용 (크레딧 0)"
    )
    parser.add_argument("--text", help="소스 텍스트(참고용; 프롬프트에는 규칙만 포함)")
    parser.add_argument("--target-lang", required=True, help="타겟 언어/시트. 예: 'DE(독일)' 또는 'German'")
    parser.add_argument("--source-lang", default="English", help="소스 언어 (기본 English)")
    parser.add_argument("--target-lang-code", help="audit 시 타겟 언어 코드(글로서리 기준)")
    parser.add_argument("--row-key", default="", help="셀 맥락 키. 예: title/button/description/disclaimer")
    parser.add_argument("--bx", action="store_true", help="BX 스타일 섹션 포함(영어 타겟)")
    parser.add_argument("--glossary", action="store_true", help="용어집 포맷 섹션 포함")
    parser.add_argument("--rag-context", help="번역 프롬프트에 주입할 RAG 사례 문자열(rag_lookup.py 결과)")
    parser.add_argument("--audit", action="store_true", help="번역 대신 검수(audit) 프롬프트 생성")
    parser.add_argument("--translated", help="audit 대상 번역문(참고용)")
    parser.add_argument("--app-root", help="app repo 경로 명시")
    parser.add_argument("--json", action="store_true", help="JSON 출력")
    args = parser.parse_args()

    try:
        res = build(args)
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2))
        else:
            print(f"❌ {e}")
        sys.exit(1)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(f"# {res['mode'].upper()} PROMPT — {res['target_lang']} (크레딧 0)")
        print(f"# glossary 맥락: {res['glossary_context_mode']} / 괄호 적용: {res['wrap_glossary_brackets']} {res['brackets']}")
        print("# ─────────────────────────────────────────────────────────")
        print(res["prompt"])


if __name__ == "__main__":
    main()
