#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
co_batch_cost_estimate.py — CO(콜롬비아) 배치 번역 전, 예상 토큰/비용을 미리 추정한다 (크레딧 0).

workbook_translate.py 에는 --dry-run 이 없으므로, 실제 LLM 호출 전에 이 스크립트로 먼저
대상 파일 수 × 토큰량을 확인하고 승인받는다. scripts/token_cost_report.py 의 접근(실제
PromptBuilder로 프롬프트를 만들고 ModelHandler.count_tokens 로 무료 계산)을 재사용하되,
스토리 2개 하드코딩 대신 임의 개수의 파일에 대해 일반화했다.

⚠️ 입력(sys+user) 토큰은 실제 프롬프트 텍스트를 만들어 정확히 센다. 출력(번역문/검수/역번역)
   토큰은 아직 생성 전이라 정확히 잴 수 없어 아래 비율로 근사한다(과거 사례 기반 추정치이며
   실제와 다를 수 있음 — 필요시 --output-ratio 등으로 조정):
     - 번역 출력 ≈ 입력 원문 토큰 × 1.0
     - 검수(AI 평가) 출력 ≈ 셀당 고정 120 tok (짧은 판정문 가정)
     - 역번역 출력 ≈ 번역 출력과 동일 근사

가격(PRICING)은 app 의 translation_web_app.model_pricing 에서 가져온다 (공식 페이지 2026-09-09
확인, standard tier). Gemini 3.6~3.8 Flash 는 2026-12-31 까지 도입가라 실행 날짜에 따라 자동 전환된다.

사용 예:
  python co_batch_cost_estimate.py --input-dir "@translation_data/@excel" \\
      --glossary runtime/glossary/latest_glossary_260803.csv
  python co_batch_cost_estimate.py --files a.xlsx b.xlsx c.xlsx
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import _app_pipeline as ap  # noqa: E402
import bootstrap as _bs  # noqa: E402

TRANSLATION_MODEL = "gemini-3.8-flash"
AUDIT_MODEL = "gpt-5.2"
SOURCE_SHEET = "US(미국)"
TARGET_SHEET = "CO(콜롬비아)"
TARGET_LANG = "Spanish_Colombia"
TARGET_LANG_CODE = "스페인어_콜롬비아"
SOURCE_LANG = "English"

# 단가는 app 의 translation_web_app.model_pricing 이 단일 출처다 (공식 페이지 2026-09-09 확인).
# app src 는 main_async 가 sys.path 에 넣은 뒤에야 import 되므로 거기서 채운다.
PRICING: dict[str, dict[str, float]] = {}
AUDIT_OUTPUT_TOK_PER_CELL = 120  # 근사치


def extract_source(xlsx_path: Path, sheet_name: str) -> dict:
    """C열 텍스트만 사용한다 — B열 행 라벨은 수식이라 openpyxl로 재저장된 파일(예:
    prep/fix 사본)에서는 캐시된 계산값이 사라진다(data_only=True로도 None). 실제 파이프라인
    (checker_service._get_row_key)은 수식 문자열을 직접 파싱해 이 문제를 피하지만, 비용
    추정에는 행 라벨 자체가 필요 없으므로 행 번호를 키로 쓴다."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    try:
        if sheet_name not in wb.sheetnames:
            return {}
        ws = wb[sheet_name]
        result = {}
        for r in range(7, 29):
            text = ws.cell(r, 3).value
            if text and str(text).strip().lower() != "x":
                result[f"row{r}"] = str(text)
        return result
    finally:
        wb.close()


async def estimate_file(pb, mh, tc, xlsx_path: Path, glossary_loaded: bool) -> dict | None:
    source_dict = extract_source(xlsx_path, SOURCE_SHEET)
    if not source_dict:
        return None

    glossary_dict = tc._get_glossary_context_as_dict(
        target_lang_code=TARGET_LANG_CODE,
        source_text=" ".join(source_dict.values()),
        row_key="description",
    ) if glossary_loaded else None

    tr_sections = pb.build_translation_prompt_sections(
        target_lang=TARGET_LANG, source_lang=SOURCE_LANG,
        row_key="description",
        glossary_context=glossary_dict or None,
    )
    tr_sys_text = "\n\n".join(s["content"] for s in tr_sections if s.get("active") and s.get("content"))
    tr_user_text = str({"source_text": source_dict, "target_language": TARGET_LANG, "glossary": glossary_dict})

    au_sections = pb.build_audit_prompt_sections(
        target_lang=TARGET_LANG, target_lang_code=TARGET_LANG_CODE,
        row_key="description", glossary_context=glossary_dict or None,
    )
    au_sys_text = "\n\n".join(s["content"] for s in au_sections if s.get("active") and s.get("content"))
    au_user_text = str({"source": source_dict, "translation": source_dict, "glossary": glossary_dict})

    (tr_sys_tok, _), (tr_user_tok, _), (au_sys_tok, _), (au_user_tok, _) = await asyncio.gather(
        mh.count_tokens(tr_sys_text, TRANSLATION_MODEL),
        mh.count_tokens(tr_user_text, TRANSLATION_MODEL),
        mh.count_tokens(au_sys_text, AUDIT_MODEL),
        mh.count_tokens(au_user_text, AUDIT_MODEL),
    )
    tr_out_tok, _ = await mh.count_tokens(" ".join(source_dict.values()), TRANSLATION_MODEL)

    n_cells = len(source_dict)
    tr_in = tr_sys_tok + tr_user_tok
    tr_cost = tr_in * PRICING[TRANSLATION_MODEL]["input"] + tr_out_tok * PRICING[TRANSLATION_MODEL]["output"]
    au_in = au_sys_tok + au_user_tok
    au_out = n_cells * AUDIT_OUTPUT_TOK_PER_CELL
    au_cost = au_in * PRICING[AUDIT_MODEL]["input"] + au_out * PRICING[AUDIT_MODEL]["output"]
    # 역번역: get_back_translation 프롬프트(짧은 지시문 + 번역 출력 텍스트) + 출력(≈ 번역 출력과 동일 근사)
    bt_in_tok = tr_out_tok + 40  # 지시문 오버헤드 근사
    bt_out_tok = tr_out_tok
    bt_cost = bt_in_tok * PRICING[TRANSLATION_MODEL]["input"] + bt_out_tok * PRICING[TRANSLATION_MODEL]["output"]

    return {
        "file": xlsx_path.name,
        "cells": n_cells,
        "tr_cost": tr_cost,
        "au_cost": au_cost,
        "bt_cost": bt_cost,
        "total": tr_cost + au_cost + bt_cost,
    }


async def main_async(files: list[Path], glossary: str | None, app_root_arg: str | None):
    app_root, _ = _bs.resolve_app_root(app_root_arg)
    if not app_root:
        raise RuntimeError("app repo를 찾지 못했습니다. --app-root 를 지정하세요.")
    src_dir = app_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    ap.maybe_reexec_with_app_venv(app_root)

    from translation_web_app.model_handler import ModelHandler
    from translation_web_app.prompt_builder import PromptBuilder
    from translation_web_app.checker_service import TranslationChecker
    from translation_web_app.model_pricing import pricing_table, pricing_note

    global PRICING
    PRICING = pricing_table([TRANSLATION_MODEL, AUDIT_MODEL])

    pb = PromptBuilder()
    mh = ModelHandler()
    tc = TranslationChecker()

    glossary_loaded = False
    if glossary:
        msg = await tc.load_glossary_from_file(glossary, TARGET_LANG_CODE)
        glossary_loaded = bool(tc.glossary)
        print(f"용어집: {msg} (loaded={glossary_loaded}, {len(tc.glossary)}개 항목)")
    else:
        print("⚠ --glossary 미지정 — 용어집 없이 추정(실제보다 입력 토큰이 적게 나올 수 있음)")

    rows = []
    for f in files:
        r = await estimate_file(pb, mh, tc, f, glossary_loaded)
        if r is None:
            print(f"  (건너뜀) {f.name}: '{SOURCE_SHEET}' 소스 텍스트 없음")
            continue
        rows.append(r)
        print(f"  {f.name}: {r['cells']}셀, 예상 ${r['total']:.4f} "
              f"(번역 ${r['tr_cost']:.4f} + 검수 ${r['au_cost']:.4f} + 역번역 ${r['bt_cost']:.4f})")

    if not rows:
        print("추정할 파일이 없습니다.")
        return

    total = sum(r["total"] for r in rows)
    total_cells = sum(r["cells"] for r in rows)
    print("─" * 60)
    print(f"대상 파일 {len(rows)}개, 총 {total_cells}셀")
    print(f"예상 총 비용: ${total:.4f}  (파일당 평균 ${total/len(rows):.4f})")
    print(f"모델: 번역={TRANSLATION_MODEL}, 검수/역번역={AUDIT_MODEL}(역번역은 번역 모델 재사용)")
    print("※ 출력 토큰(번역문·검수 판정·역번역)은 근사치입니다. 정확한 값은 실제 실행 후에만 알 수 있습니다.")
    print(pricing_note())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", nargs="+", help="추정할 .xlsx 파일 목록")
    p.add_argument("--input-dir", help="이 폴더의 모든 *.xlsx 추정 (~$ 잠금파일 제외)")
    p.add_argument("--glossary", help="용어집 CSV 경로(입력 토큰 정확도를 위해 지정 권장)")
    p.add_argument("--app-root", help="app repo 경로 명시")
    args = p.parse_args()

    if args.files:
        files = [Path(f).expanduser() for f in args.files]
    elif args.input_dir:
        base = Path(args.input_dir).expanduser()
        files = sorted(p2 for p2 in base.glob("*.xlsx") if not p2.name.startswith("~$"))
    else:
        print("❌ --files 또는 --input-dir 중 하나는 지정해야 합니다.")
        sys.exit(2)

    asyncio.run(main_async(files, args.glossary, args.app_root))


if __name__ == "__main__":
    main()
