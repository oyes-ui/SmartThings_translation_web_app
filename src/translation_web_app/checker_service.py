# -*- coding: utf-8 -*-
"""
backend/checker_service.py
Refactored for Web API usage (FastAPI + SSE)
"""

import openpyxl
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
import os
from google import genai
from google.genai import types
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import json
import re
import io
import urllib.parse
from translation_web_app.model_handler import ModelHandler
from translation_web_app.prompt_builder import PromptBuilder
from translation_web_app.rules_loader import get_rules
from translation_web_app.report_builder import build_front_matter, render_finding, render_report
from translation_web_app.glossary_checks import GlossaryChecker

_GRADE_KR = {"Excellent": "우수", "Good": "양호", "Needs Revision": "수정 필요"}

# RAG 연동 (DB가 없우면 graceful fallback)
try:
    from translation_web_app.rag_retriever import get_retriever as _get_rag_retriever
except ImportError:
    _get_rag_retriever = None

# API Keys from Environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class TranslationChecker:
    """
    Excel Translation Checker (Gemini Single Model) - Service Version
    """

    def __init__(
        self,
        model_name: str = "gpt-5.4-mini",
        max_concurrency: int = 10,
        short_text_whitelist=None,
        skip_llm_when_glossary_mismatch: bool = False,
        no_backtranslation: bool = False,
        backtranslation_lang: str = None,
        backtranslation_sheet: str = None,
        gemini_api_key: str = None,
        openai_api_key: str = None,
        audit_reasoning_effort: str = None,
        audit_thinking_budget: int = None,
    ):
        # API Setup
        self.model_name = model_name
        self.audit_reasoning_effort = audit_reasoning_effort
        self.audit_thinking_budget = audit_thinking_budget
        
        # Concurrency
        if max_concurrency < 1:
            max_concurrency = 1
        self.max_concurrency = max_concurrency
        self._sem = asyncio.Semaphore(self.max_concurrency)
            
        self.no_backtranslation = no_backtranslation
        # 역번역 참조 언어를 실제 번역 source_lang과 분리하고 싶을 때 사용(예: 소스가
        # 영어라도 한국어 리뷰어를 위해 항상 한국어로 역번역). None이면 기존 동작(=source_lang) 유지.
        self.backtranslation_lang = backtranslation_lang
        # 역번역 결과를 같은 시트의 인접 열이 아니라 별도 시트(같은 좌표)에 기록하고 싶을 때
        # 시트명을 지정한다. None이면 역번역 결과를 Excel에 기록하지 않는다(보고서에만 남음).
        self.backtranslation_sheet = backtranslation_sheet

        # Whitelist
        default_whitelist = {"ok", "on", "off", "ai", "5g", "go", "up", "usb", "nfc"}
        if short_text_whitelist:
            if isinstance(short_text_whitelist, str):
                extra = {x.strip().lower() for x in short_text_whitelist.split(",") if x.strip()}
            else:
                extra = {str(x).strip().lower() for x in short_text_whitelist}
            default_whitelist |= extra
        self.short_text_whitelist = default_whitelist

        self.skip_llm_when_glossary_mismatch = skip_llm_when_glossary_mismatch

        # Integrated components
        self.model_handler = ModelHandler(gemini_api_key=gemini_api_key, openai_api_key=openai_api_key)
        self.prompt_builder = PromptBuilder()

        # Glossary loading + matching/casing/bracket/brand checks — moved to
        # glossary_checks.py (2026-08-03). Thin wrappers below preserve the original
        # method names/attributes for external callers that reach into them directly.
        self._glossary_checker = GlossaryChecker(self.prompt_builder)

        # RAG Retriever (오프라인 전용, DB 없으면 None)
        self.rag_retriever = None
        if _get_rag_retriever:
            try:
                self.rag_retriever = _get_rag_retriever()
                if not self.rag_retriever.is_available():
                    self.rag_retriever = None
            except Exception:
                self.rag_retriever = None
    

    # ----------------- Glossary (delegates to GlossaryChecker) -----------------
    # 실제 구현은 glossary_checks.py 로 옮겨졌다(2026-08-03). 이름·시그니처를 그대로 둔
    # 얇은 위임만 남긴다 — 외부 스크립트 여러 곳(workbook_incremental_highlight.py,
    # co_batch_cost_estimate.py, scripts/token_cost_report*.py, main.py 데모 엔드포인트)이
    # 이 "private" 메서드와 .glossary 속성을 직접 참조하기 때문에 이름을 바꿀 수 없다.
    @property
    def glossary(self):
        return self._glossary_checker.glossary

    @glossary.setter
    def glossary(self, value):
        self._glossary_checker.glossary = value

    @property
    def glossary_re(self):
        return self._glossary_checker.glossary_re

    @glossary_re.setter
    def glossary_re(self, value):
        self._glossary_checker.glossary_re = value

    @property
    def glossary_map(self):
        return self._glossary_checker.glossary_map

    @glossary_map.setter
    def glossary_map(self, value):
        self._glossary_checker.glossary_map = value

    @property
    def glossary_headers(self):
        return self._glossary_checker.glossary_headers

    @glossary_headers.setter
    def glossary_headers(self, value):
        self._glossary_checker.glossary_headers = value

    @property
    def source_lang_code(self):
        return self._glossary_checker.source_lang_code

    @source_lang_code.setter
    def source_lang_code(self, value):
        self._glossary_checker.source_lang_code = value

    async def load_glossary_from_file(self, file_path: str, source_lang_code: str):
        return await self._glossary_checker.load_glossary_from_file(file_path, source_lang_code)

    def load_activation_manifest(self, file_path: str | None) -> None:
        """Occurrence activation is a per-story overlay, never a replacement glossary."""
        self._glossary_checker.load_activation_manifest(file_path)

    def resolve_constraints(self, source_text: str, target_lang_code: str, **kwargs):
        return self._glossary_checker.resolve_constraints(source_text, target_lang_code, **kwargs)

    def validate_constraints(self, target_text: str, constraint_card: dict):
        return self._glossary_checker.validate_constraints(target_text, constraint_card)

    def is_term_active_for_occurrence(
        self, source_term: str, rule: str, story: str | None = None, cell: str | None = None
    ) -> bool:
        return self._glossary_checker.is_term_active_for_occurrence(source_term, rule, story=story, cell=cell)

    def _get_target_val(self, targets: dict, lang_code: str) -> str | None:
        return self._glossary_checker._get_target_val(targets, lang_code)

    def _extract_glossary_target_terms(self, glossary_context_or_terms) -> list[str]:
        return self._glossary_checker._extract_glossary_target_terms(glossary_context_or_terms)

    def _restore_glossary_target_casing(self, target_text: str, glossary_context_or_terms) -> str:
        return self._glossary_checker._restore_glossary_target_casing(target_text, glossary_context_or_terms)

    def _compile_glossary_re(self):
        return self._glossary_checker._compile_glossary_re()

    def _get_relevant_glossary_terms(self, source_text: str):
        return self._glossary_checker._get_relevant_glossary_terms(source_text)

    def _build_glossary_lines_for_code(self, target_lang_code: str, source_text: str = None):
        return self._glossary_checker._build_glossary_lines_for_code(target_lang_code, source_text)

    def _get_glossary_context_as_dict(self, target_lang_code: str, source_text: str | None = None,
                                       skip_deactivated: bool = False, row_key: str = ""):
        return self._glossary_checker._get_glossary_context_as_dict(
            target_lang_code, source_text=source_text, skip_deactivated=skip_deactivated, row_key=row_key
        )

    def _precheck_glossary_mismatch(self, source_text: str, target_text: str, target_lang_code: str):
        return self._glossary_checker._precheck_glossary_mismatch(source_text, target_text, target_lang_code)

    def _check_glossary_casing(self, source_text: str, target_text: str, target_lang_code: str):
        return self._glossary_checker._check_glossary_casing(source_text, target_text, target_lang_code)

    def _get_navigation_path_spans(self, target_text: str, target_lang: str = "") -> list:
        return self._glossary_checker._get_navigation_path_spans(target_text, target_lang)

    def _is_inside_span(self, start: int, end: int, spans: list[tuple[int, int]]) -> bool:
        return self._glossary_checker._is_inside_span(start, end, spans)

    def _check_glossary_brackets(self, source_text: str, target_text: str, target_lang_code: str,
                                  target_lang: str, row_key: str = ""):
        return self._glossary_checker._check_glossary_brackets(
            source_text, target_text, target_lang_code, target_lang, row_key=row_key
        )

    def _split_context_terms(self, glossary_context):
        return self._glossary_checker._split_context_terms(glossary_context)

    def _unwrap_glossary_brackets(self, text: str, terms, spans=None, pairs=None) -> str:
        if pairs is None:
            return self._glossary_checker._unwrap_glossary_brackets(text, terms, spans=spans)
        return self._glossary_checker._unwrap_glossary_brackets(text, terms, spans=spans, pairs=pairs)

    def _strip_glossary_brackets_by_policy(self, target_text: str, glossary_context, row_key: str = "") -> str:
        return self._glossary_checker._strip_glossary_brackets_by_policy(target_text, glossary_context, row_key)

    def _strip_title_button_glossary_brackets(self, target_text: str, glossary_context, row_key: str = "") -> str:
        return self._glossary_checker._strip_title_button_glossary_brackets(target_text, glossary_context, row_key)

    def _build_glossary_word_set(self) -> set:
        return self._glossary_checker._build_glossary_word_set()

    def _brand_like_target_terms(self, source_text: str) -> list[str]:
        return self._glossary_checker._brand_like_target_terms(source_text)

    def _check_brand_concatenation(self, source_text: str, target_text: str, target_lang_code: str,
                                    target_lang: str = "") -> list:
        return self._glossary_checker._check_brand_concatenation(source_text, target_text, target_lang_code, target_lang)

    def _check_disclaimer_linebreak(self, source_text: str, target_text: str, row_key: str = "") -> list:
        return self._glossary_checker._check_disclaimer_linebreak(source_text, target_text, row_key)

    def _check_nav_path_bracket_leak(self, target_text: str, row_key: str = "") -> list:
        return self._glossary_checker._check_nav_path_bracket_leak(target_text, row_key)

    def _analyze_sentence_case(self, target_text: str, target_lang: str, glossary_terms=None):
        return self._glossary_checker._analyze_sentence_case(target_text, target_lang, glossary_terms)

    def _get_row_key(self, worksheet, row_idx: int) -> str:
        cell = worksheet[f"B{row_idx}"]
        value = cell.value
        if value is None:
            return ""
        raw = str(value).strip()
        if not raw:
            return ""
        if not raw.startswith("="):
            return raw

        # Best-effort evaluator for the project row-key formula pattern:
        # ="//section_"&RIGHT($C$5, 3)&"_1_button"
        story_id = ""
        c5 = worksheet["C5"].value
        if c5 is not None:
            digits = re.findall(r"\d+", str(c5))
            if digits:
                story_id = digits[-1][-3:].zfill(3)

        formula_body = raw[1:]
        parts = []
        for token in formula_body.split("&"):
            token = token.strip()
            quoted = re.fullmatch(r'"(.*)"', token)
            if quoted:
                parts.append(quoted.group(1))
                continue
            if "RIGHT" in token.upper() and story_id:
                parts.append(story_id)

        resolved = "".join(parts).strip()
        return resolved if resolved.startswith("//") else raw

    # ----------------- Report Rendering -----------------
    def _render_report(
        self,
        *,
        title,
        source_file_id,
        findings,
        summary_lines,
        workflow="app_review",
        translation_model=None,
    ):
        """Single entry point so every pipeline emits the same Markdown shape."""
        report_id = f"review-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return render_report(
            title=title,
            front_matter=build_front_matter(
                report_id=report_id,
                source_file_id=os.path.basename(str(source_file_id or "")) or "unknown",
                workflow=workflow,
                translation_model=translation_model,
                audit_model=self.model_name,
            ),
            findings=findings,
            summary_lines=summary_lines,
            usage_report=self.model_handler.get_usage_report(),
        )

    def _render_highlight_report(self, *, source_file_id, out_excel_path, summary_lines, detail_lines):
        """Highlight-only runs have no per-cell audit, so they carry no findings."""
        body = []
        if detail_lines:
            body += ["### 주요 용어집 준수여부 점검 알림", "", "```text",
                     "\n".join(detail_lines).strip(), "```", ""]
        report = self._render_report(
            title="하이라이트 전용 검수 보고서",
            source_file_id=source_file_id,
            findings=body,
            summary_lines=[*summary_lines, f"하이라이트본: {os.path.basename(out_excel_path)}"],
            workflow="highlight_only",
        )
        return report

    # ----------------- Excel Loader -----------------
    def load_excel_data(self, source_path, target_path, cell_range="A:Z", selected_sheets=None, log_func=None, source_sheet_name=None):
        all_data = []
        if log_func: log_func(f"데이터 추출 범위: {cell_range}")
        try:
            wb_source = openpyxl.load_workbook(source_path, data_only=True)
            wb_target = openpyxl.load_workbook(target_path, data_only=True)
        except Exception as e:
            raise Exception(f"Excel load error: {e}")

        source_order = wb_source.sheetnames
        target_set = set(wb_target.sheetnames)
        common_sheets = [s for s in source_order if s in target_set]

        if not common_sheets:
            raise Exception("No matching sheets found between Source and Target files.")

        if selected_sheets:
            target_sheets = [s for s in selected_sheets if s in common_sheets]
            if not target_sheets:
                raise Exception(f"Selected sheets {selected_sheets} not found in common sheets.")
        else:
            target_sheets = common_sheets

        processed_sheets = []

        for sheet_name in target_sheets:
            # Determine source sheet for this item
            src_sheet = source_sheet_name if source_sheet_name else sheet_name
            
            if src_sheet not in wb_source.sheetnames or sheet_name not in wb_target.sheetnames:
                if log_func: log_func(f"⚠ [{sheet_name}] 시트가 원본 파일({src_sheet}) 또는 대상 파일에 없습니다. 건너뜁니다.")
                continue
            
            ws_source = wb_source[src_sheet]
            ws_target = wb_target[sheet_name]
            story_digits = re.findall(r"\d+", str(ws_source["C5"].value or ""))
            story_id = story_digits[-1][-3:].zfill(3) if story_digits else None
            
            extracted_count = 0
            extracted_coords = set()
            # 쉼표로 구분된 범위를 나누고, 개별 셀(c16)이나 범위(c10:c20) 모두 지원
            range_parts = [r.strip() for r in cell_range.split(',') if r.strip()]
            
            for current_range in range_parts:
                try:
                    # Handle full range like "A:Z" or specific "A1:C10"
                    src_rows = ws_source[current_range]
                    tgt_rows = ws_target[current_range]
                    
                    # Normalize to list of rows if single cell
                    if not isinstance(src_rows, tuple) and not isinstance(src_rows, list): 
                        # it might be a single cell if range is "A1"
                        src_rows = ((src_rows,),)
                        tgt_rows = ((tgt_rows,),)

                    for source_row, target_row in zip(src_rows, tgt_rows):
                        for s_cell, t_cell in zip(source_row, target_row):
                            if s_cell.coordinate in extracted_coords: 
                                continue
                            extracted_coords.add(s_cell.coordinate)
                            
                            s_val = str(s_cell.value).strip() if s_cell.value is not None else ""
                            t_val = str(t_cell.value).strip() if t_cell.value is not None else ""
                            
                            if s_val and t_val and s_val.lower() != "x" and t_val.lower() != "x":
                                # Extract row_key from column B of the same row
                                row_idx = s_cell.row
                                row_key = self._get_row_key(ws_source, row_idx)
                                
                                all_data.append({
                                    "cell_ref": s_cell.coordinate,
                                    "sheet_name": sheet_name,
                                    "source": s_val,
                                    "target": t_val,
                                    "row_key": row_key,
                                    "story_id": story_id,
                                })
                                extracted_count += 1
                except Exception as e:
                    print(f"Error accessing range {current_range} in {sheet_name}: {e}")
                    continue
            
            if extracted_count > 0:
                processed_sheets.append(sheet_name)
                if log_func: log_func(f"✓ [{sheet_name}] {extracted_count}개 항목 추출 완료")
            else:
                if log_func: log_func(f"⚠ [{sheet_name}] 해당 범위({cell_range})에 유효한 데이터가 없습니다.")
        
        return all_data, processed_sheets

    # ----------------- Helpers -----------------
    async def _with_semaphore(self, coro):
        async with self._sem:
            return await coro

    async def _rag_lookup(self, source_text, sheet_title, source_lang, rag_identity_match=True, n_results=2):
        """RAG retrieve() 의 가용성 체크·동기 호출 오프로드·예외 처리를 한 곳에 모은다.

        여러 호출부(감사 리포트용 process_item, 프롬프트 삽입용 파이프라인 워커 2곳)가
        이 retrieve+try/except 블록을 그대로 복붙해왔다 — 이후 결과를 어떻게 포맷팅하는지는
        호출부마다 달라서(리포트 text/JSON vs 프롬프트 문자열) 포맷팅은 각자 하고, 여기서는
        조회 자체만 공통화한다. 반환: (results: list[dict], error_message: str | None).
        """
        if not (getattr(self, "rag_retriever", None) and self.rag_retriever.is_available()):
            return [], None
        try:
            results = await asyncio.to_thread(
                self.rag_retriever.retrieve,
                source_text,
                sheet_title,
                source_lang=source_lang,
                n_results=n_results,
                exclude_same_source=False,
                identity_match_enabled=rag_identity_match,
            )
            return results, None
        except Exception as e:
            return [], str(e)

    async def _fetch_rag_prompt_context(self, source_text, sheet_title, source_lang, coord, rag_identity_match=True):
        """번역 프롬프트에 주입할 RAG 컨텍스트 문자열을 만든다 (파이프라인 셀 워커 2곳 공용).

        반환: (rag_context_str: str | None, log_messages: list[str]).
        """
        results, err = await self._rag_lookup(source_text, sheet_title, source_lang, rag_identity_match)
        if err:
            return None, [f"[RAG] 검색 오류: {err}"]
        if not results:
            return None, []
        rag_context_str = await asyncio.to_thread(
            self.rag_retriever.format_for_prompt,
            source_text,
            sheet_title,
            source_lang=source_lang,
            n_results=2,
            identity_match_enabled=rag_identity_match,
        )
        return rag_context_str, [f"[RAG] 유사 사례 {len(results)}건 적용됨 (시트: {sheet_title}, 셀: {coord})"]

    async def _run_llm_translation(self, text, target_lang, model_name="gemini-3.6-flash", bx_style_on=False, glossary_context=None, rag_context=None, row_key="", source_lang="English", rag_identity_match=True, target_lang_code="", thinking_budget: int | None = None, constraint_card: dict | None = None):
        """
        내부 전용 번역 메서드: JSON 프롬프트 생성 및 LLM 호출을 담당합니다.
        """
        # 입력 데이터 구조화
        input_data = {
            "context_key": row_key,
            "source_text": text,
            "target_language": target_lang,
            "glossary": glossary_context if isinstance(glossary_context, dict) else {},
            "formatting": self.prompt_builder.build_input_formatting(target_lang, row_key)
        }

        # RAG 예시 주입: 유사 번역 사례 최대 2건을 시스템 프롬프트에 첨부
        # format_for_prompt 내부의 Gemini 임베딩 호출이 동기(sync)라, 이벤트루프에서
        # 직접 부르면 그 응답이 올 때까지 서버 전체가 멈춘다 — 스레드로 오프로드.
        if self.rag_retriever and not rag_context:
            try:
                rag_context = await asyncio.to_thread(
                    self.rag_retriever.format_for_prompt,
                    text,
                    target_lang,
                    source_lang=source_lang,
                    n_results=2,
                    identity_match_enabled=rag_identity_match
                )
                if rag_context:
                    pass # PromptBuilder handles injection via build_translation_prompt
            except Exception:
                pass  # RAG 실패해도 번역은 정상 진행

        system_prompt = self.prompt_builder.build_translation_prompt(
            target_lang=target_lang,
            source_lang=source_lang,
            bx_style_on=bx_style_on,
            rag_context=rag_context,
            row_key=row_key,
            glossary_context=glossary_context,
            hard_constraint_card=constraint_card,
        )

        try:
            prompt = json.dumps(input_data, ensure_ascii=False, indent=2)
            response_data = await self.model_handler.generate_content(
                prompt,
                model_name=model_name,
                system_instruction=system_prompt,
                response_json=True,
                thinking_budget=thinking_budget,
            )

            if isinstance(response_data, dict):
                translation = response_data.get("translation", str(response_data))
            else:
                translation = str(response_data)
            translation = self._strip_glossary_brackets_by_policy(translation, glossary_context, row_key)
            translation = self._restore_glossary_target_casing(translation, glossary_context)
            if constraint_card:
                verdict = self.validate_constraints(translation, constraint_card)
                if verdict["status"] == "blocked":
                    raise ValueError(f"resolver가 번역 결과를 차단했습니다: {verdict['blocked']}")
            return translation
        except Exception as e:
            return f"[번역 오류] {str(e)}"


    # ----------------- LLM Calls -----------------
    async def check_with_llm_qa(self, source_text, target_text, source_lang, target_lang, target_lang_code,
                                row_key: str = "", constraint_card: dict | None = None,
                                pending_activation_candidates: list[dict] | None = None):
        glossary_dict = self._get_glossary_context_as_dict(
            target_lang_code,
            source_text=source_text,
            skip_deactivated=True,
            row_key=row_key,
        )
        
        # QA용 구조화된 프롬프트 데이터
        input_data = {
            "source": {"lang": source_lang, "text": source_text},
            "translation": {"lang": f"{target_lang}/{target_lang_code}", "text": target_text},
            "glossary": glossary_dict,
            "context_key": row_key
        }

        prompt = f"""당신은 전문 번역 검수 전문가입니다. 아래 JSON 데이터를 분석하여 상세한 검수 결과를 반환하세요.

[Input Data]
{json.dumps(input_data, ensure_ascii=False, indent=2)}
"""
        system_instruction = self.prompt_builder.build_audit_prompt(
            source_lang=source_lang,
            target_lang=target_lang,
            target_lang_code=target_lang_code,
            row_key=row_key,
            glossary_context=glossary_dict,
            hard_constraint_card=constraint_card,
        )
        try:
            # ModelHandler가 JSON 모드를 지원하므로 딕셔너리로 바로 받을 수 있음
            response = await self.model_handler.generate_content(
                prompt,
                model_name=self.model_name,
                system_instruction=system_instruction,
                response_json=True,
                reasoning_effort=self.audit_reasoning_effort,
                thinking_budget=self.audit_thinking_budget,
            )
            
            if isinstance(response, dict) and response.get("error") == "parsing_failed":
                 err_msg = f"[파싱/서버 오류]: Gemini가 유효한 JSON을 반환하지 않았거나 서버가 응답하지 않습니다. (원문: {str(response.get('original_text', ''))[:100]}...)"
                 eval_json = json.dumps({"evaluation": [{"category": "에러", "comment": err_msg}], "grade": "Needs Revision", "suggested_fix": ""}, ensure_ascii=False)
                 return (err_msg, eval_json)

            if isinstance(response, str):
                 eval_text = f"[QA 분석 결과]\n{response}"
                 eval_json = json.dumps({"evaluation": [{"category": "결과", "comment": response}], "grade": "Needs Revision", "suggested_fix": ""}, ensure_ascii=False)
                 return (eval_text, eval_json)
            
            # 1. Build beautiful human-readable text
            eval_list = response.get("evaluation", [])
            lines = []
            for item in eval_list:
                cat = item.get("category", "")
                com = item.get("comment", "")
                if cat and com:
                   lines.append(f"- {cat}: {com}")
            
            eval_text = "\n".join(lines) if lines else "검수 결과 없음."
            grade = response.get("grade", "")
            grade_label = _GRADE_KR.get(grade, grade)
            grade_desc = dict(get_rules().doc("audit").labelled("grade")).get(grade, "")
            if grade_label:
                suffix = f" — {grade_desc}" if grade_desc else ""
                eval_text += f"\n\n최종 평가: {grade_label}{suffix}"
            raw_suggested_fix = response.get("suggested_fix", "")
            suggestion_validation = None
            suggested_fix = raw_suggested_fix
            if raw_suggested_fix and constraint_card is not None:
                suggestion_validation = self._glossary_checker.validate_audit_suggestion(
                    raw_suggested_fix, constraint_card,
                    glossary_context=glossary_dict, row_key=row_key,
                )
                violations = suggestion_validation.get("violations") or []
                pending_terms = {str(item.get("source_term", ""))
                                 for item in (pending_activation_candidates or [])
                                 if not item.get("confirmed")}
                activation_hits = [item for item in violations
                                   if item.get("reason") == "missing_glossary_target"
                                   and str(item.get("source_term", "")) in pending_terms]
                if activation_hits and len(activation_hits) == len(violations) and not suggestion_validation.get("review"):
                    suggestion_validation["disposition"] = "glossary_activation_review"
                    suggestion_validation["activation_candidates"] = activation_hits
                elif suggestion_validation["status"] != "pass":
                    suggestion_validation["disposition"] = "invalidated_by_hard_constraint"
                if suggestion_validation["status"] == "pass":
                    suggested_fix = suggestion_validation["normalized"]
                    eval_text += f"\n\n[수정안 제안]:\n{suggested_fix}"
                else:
                    suggested_fix = ""
                    reasons = suggestion_validation.get("violations") or suggestion_validation.get("review") or []
                    eval_text += "\n\n[수정안 resolver 검토 필요]:\n" + json.dumps(
                        reasons, ensure_ascii=False)
            elif raw_suggested_fix:
                eval_text += f"\n\n[수정안 제안]:\n{raw_suggested_fix}"

            # 2. Extract JSON payload
            eval_json = json.dumps({
                "evaluation": eval_list,
                "grade": grade or "Needs Revision",
                "suggested_fix": suggested_fix,
                "raw_suggested_fix": raw_suggested_fix,
                "suggestion_validation": suggestion_validation,
            }, ensure_ascii=False)
            
            return (eval_text, eval_json)
        except Exception as e:
            err_text = f"[QA 오류]: {str(e)}"
            err_json = json.dumps({"evaluation": [{"category": "에러", "comment": str(e)}], "grade": "Needs Revision", "suggested_fix": ""}, ensure_ascii=False)
            return (err_text, err_json)






    async def get_back_translation(self, target_text, target_lang, source_lang):
        prompt = (
            f"다음 {target_lang} 텍스트를 {source_lang}으로 다시 번역해주세요. "
            f"오직 번역된 텍스트만 제공해야 합니다. 다른 설명이나 텍스트는 포함하지 마세요.\n\n{target_text}"
        )
        try:
            return await self.model_handler.generate_content(
                prompt, model_name=self.model_name,
                reasoning_effort=self.audit_reasoning_effort,
                thinking_budget=self.audit_thinking_budget,
            )
        except Exception as e:
            return f"[역번역 오류]: {e}"

    def _write_backtranslation_cell(self, ws, coord: str, back_translation) -> None:
        """역번역 결과를 별도 시트(self.backtranslation_sheet)의 같은 좌표에 기록한다.

        self.backtranslation_lang 과 self.backtranslation_sheet 이 모두 설정된 경우에만
        동작(opt-in) — 기존 24개 언어 시트나 웹 UI 경로(항상 둘 다 None)에는 영향을 주지 않는다.
        대상 시트는 번역 시트와 동일한 파일 안에서 미리 준비돼 있어야 한다
        (workbook_add_target_sheet.py --backtranslation-sheet 참고); 없으면 조용히 건너뛴다
        (배치 전체를 중단시키지 않도록).
        """
        if not self.backtranslation_lang or not self.backtranslation_sheet or not back_translation:
            return
        wb = ws.parent
        if self.backtranslation_sheet not in wb.sheetnames:
            return
        wb[self.backtranslation_sheet][coord] = back_translation

    def _apply_rich_text(self, text: str, keywords: list, base_font=None):
        """
        텍스트 내의 키워드를 파란색으로 하이라이트하되, 나머지 텍스트는 base_font의 스타일을 유지합니다.
        """
        if not text:
            return ""
        if not keywords:
            return text # 키워드가 없으면 일반 문자열 반환 (교체하지 않음으로써 기존 스타일 보존)
            
        sorted_keywords = sorted([k.strip() for k in keywords if k.strip()], key=len, reverse=True)
        if not sorted_keywords:
            return text

        # 정규표현식으로 키워드 위치 찾기
        pattern = '|'.join(re.escape(k) for k in sorted_keywords)
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if not matches:
            return text

        # 베이스 폰트 정보 추출 (없으면 기본값)
        # InlineFont는 Font와 달리 'name' 대신 'rFont' 사용
        font_params = {}
        if base_font:
            font_params = {
                "rFont": base_font.name,
                "sz": base_font.sz,
                "b": base_font.b,
                "i": base_font.i,
                "u": base_font.u,
                "strike": base_font.strike,
                "family": base_font.family,
                "charset": base_font.charset,
                "outline": base_font.outline,
                "shadow": base_font.shadow,
                "condense": base_font.condense,
                "extend": base_font.extend,
                "vertAlign": base_font.vertAlign,
                "scheme": base_font.scheme,
            }

        def get_font(is_keyword=False):
            params = font_params.copy()
            if is_keyword:
                params["color"] = "0000FF"  # 키워드만 파란색 명시, 나머지는 셀 기본 색상 상속
            # is_keyword=False 시 color 미설정 → Excel 셀 레벨 스타일에서 색상 상속
            return InlineFont(**params)


        parts = []
        last_end = 0
        for m in matches:
            start, end = m.span()
            if start > last_end:
                parts.append([text[last_end:start], get_font(False)])
            parts.append([text[start:end], get_font(True)])
            last_end = end

        if last_end < len(text):
            parts.append([text[last_end:], get_font(False)])

        rt = CellRichText()
        for segment_text, segment_font in self._fold_bare_whitespace_runs(parts):
            rt.append(TextBlock(text=segment_text, font=segment_font))
        return rt

    @staticmethod
    def _fold_bare_whitespace_runs(blocks):
        """Fold whitespace-only rich-text segments into an adjacent run.

        openpyxl's ``whitespace()`` helper marks a run ``xml:space="preserve"`` only when
        it holds BOTH whitespace and non-whitespace text (``stripped and text != stripped``).
        A run that is ONLY whitespace therefore ships unprotected, and Excel trims it on
        open — dropping the space (gluing consecutive highlighted terms, "Galaxy Device"
        -> "GalaxyDevice") and, worse, collapsing the neighbouring run's colour across the
        merge. Merging every whitespace-only segment into a neighbour guarantees no bare
        space run is ever emitted; the character stream is unchanged. ``blocks`` is a list
        of ``[text, font]`` (font may be None for a plain segment).
        """
        folded: list = []
        for seg_text, seg_font in blocks:
            if seg_text == "":
                continue
            if folded and seg_text.strip() == "":
                folded[-1][0] += seg_text          # attach to preceding run
            else:
                folded.append([seg_text, seg_font])
        if len(folded) >= 2 and folded[0][0].strip() == "":
            folded[1][0] = folded[0][0] + folded[1][0]   # leading space -> following run
            folded.pop(0)
        return folded

    @classmethod
    def _harden_richtext_cell(cls, value):
        """Return ``value`` with every bare whitespace-only run folded away.

        Accepts a :class:`CellRichText` — freshly built by :meth:`_apply_rich_text` or
        loaded from a workbook that Excel / an earlier pipeline pass re-segmented into a
        standalone space run. Anything else is returned unchanged.
        """
        if not isinstance(value, CellRichText):
            return value
        blocks = []
        for part in value:
            if isinstance(part, TextBlock):
                blocks.append([part.text or "", part.font])
            else:
                blocks.append([str(part), None])
        if not any(text and text.strip() == "" for text, _ in blocks):
            return value  # already free of bare whitespace runs
        rt = CellRichText()
        for seg_text, seg_font in cls._fold_bare_whitespace_runs(blocks):
            rt.append(seg_text if seg_font is None else TextBlock(text=seg_text, font=seg_font))
        return rt

    @classmethod
    def _harden_workbook_whitespace_runs(cls, wb) -> int:
        """Fold bare whitespace-only rich-text runs across every cell before saving.

        :meth:`_apply_rich_text` never emits such runs, but a workbook loaded after an
        Excel round-trip or an earlier pipeline pass can still carry them in cells this
        run did not rewrite. Running this immediately before ``wb.save`` keeps the
        delivered file free of the fragile structure regardless of how a cell got its
        rich text. Returns the number of cells repaired.
        """
        hardened = 0
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    value = cell.value
                    if isinstance(value, CellRichText):
                        new_value = cls._harden_richtext_cell(value)
                        if new_value is not value:
                            cell.value = new_value
                            hardened += 1
        return hardened

    @staticmethod
    def _cell_text_for_highlighting(value) -> str:
        """Return a cell's complete character stream for rich-text reconstruction."""
        return "" if value is None else str(value)

    @staticmethod
    def _verify_saved_cell_text(workbook_path: str, expected: dict[tuple[str, str], str]) -> None:
        """Reject a generated workbook when saving changed any produced character."""
        check_wb = openpyxl.load_workbook(workbook_path, data_only=False, rich_text=True)
        mismatches = []
        for (sheet_name, coord), expected_text in expected.items():
            actual_text = str(check_wb[sheet_name][coord].value)
            if actual_text != expected_text:
                mismatches.append(
                    f"{sheet_name}!{coord}: expected={expected_text!r}, actual={actual_text!r}"
                )
                if len(mismatches) >= 10:
                    break
        check_wb.close()
        if mismatches:
            raise RuntimeError("Excel 저장 후 텍스트 무결성 검증 실패: " + " | ".join(mismatches))

    # ----------------- Process Single Item -----------------
    async def process_item(self, item, source_lang, default_target_lang, sheet_lang_map, default_target_lang_code, rag_identity_match=True):
        cell_ref = item["cell_ref"]
        sheet_name = item["sheet_name"]
        source = item["source"]
        target = item["target"]
        
        tgt_lang = sheet_lang_map.get(sheet_name, {}).get("lang", default_target_lang)
        tgt_code = sheet_lang_map.get(sheet_name, {}).get("code", default_target_lang_code)
        
        # Skip detection
        is_placeholder = (
            source.strip().lower() == "x" or
            target.strip().lower() == "x" or
            (len(source) <= 2 and len(target) <= 2 and source.lower() == target.lower()) or
            (not source.strip() and not target.strip()) or
            (source.strip().lower() == target.strip().lower() and len(source.strip()) < 10)
        )
        if source.strip().lower() in self.short_text_whitelist or target.strip().lower() in self.short_text_whitelist:
            is_placeholder = False
        
        row_key = item.get("row_key", "")
        story = item.get("story_id") or getattr(self, "_inspection_story_id", None)
        constraint_card = self.resolve_constraints(
            source, tgt_code, row_key=row_key, story=story, cell=cell_ref,
        )
        pending_activation_candidates = self._glossary_checker.pending_inactive_candidates(
            source, story=story or "", cell=cell_ref, target_lang_code=tgt_code, target_text=target)
        constraint_validation = self.validate_constraints(target, constraint_card)
        pre_mismatch = self._precheck_glossary_mismatch(source, target, tgt_code)
        # A deterministic lexical/bracket violation is never delegated to the LLM.
        skip_llm = (self.skip_llm_when_glossary_mismatch and bool(pre_mismatch)) or constraint_validation["status"] == "blocked"
        
        relevant_terms_for_case = self._get_relevant_glossary_terms(source)
        glossary_targets_for_case = []
        for s_term in relevant_terms_for_case:
            meta = self.glossary.get(s_term)
            if not meta:
                continue
            target_val = self._get_target_val(meta["targets"], tgt_code)
            if target_val:
                glossary_targets_for_case.append(target_val)

        case_report, simple_case_fix = self._analyze_sentence_case(target, tgt_lang, glossary_targets_for_case)
        glossary_case_issues = self._check_glossary_casing(source, target, tgt_code)
        glossary_bracket_issues = self._check_glossary_brackets(source, target, tgt_code, tgt_lang, row_key=row_key)
        brand_concat_issues = self._check_brand_concatenation(source, target, tgt_code, tgt_lang)
        disclaimer_linebreak_issues = self._check_disclaimer_linebreak(source, target, row_key)
        nav_bracket_leak_issues = self._check_nav_path_bracket_leak(target, row_key)

        # Construct Partial Report Sections
        case_section = "대소문자 하드룰(문장형) 점검:\n" + case_report if case_report else "별도 지적 사항 없음."
        if simple_case_fix:
            case_section += f"\n\n[단순 규칙 기반 문장형 변환안]:\n{simple_case_fix}"

        glossary_parts = []
        if pre_mismatch: glossary_parts.append("용어집 사전 감지:\n- " + "\n- ".join(pre_mismatch))
        if glossary_case_issues: glossary_parts.append("용어집 대소문자 표기 점검:\n" + "\n".join(f"- {msg}" for msg in glossary_case_issues))
        if glossary_bracket_issues: glossary_parts.append("용어집 괄호 규정 점검:\n" + "\n".join(f"- {msg}" for msg in glossary_bracket_issues))
        if brand_concat_issues: glossary_parts.append("브랜드 띄어쓰기 점검:\n" + "\n".join(f"- {msg}" for msg in brand_concat_issues))
        if disclaimer_linebreak_issues: glossary_parts.append("디스클레이머 줄바꿈 서식 점검:\n" + "\n".join(f"- {msg}" for msg in disclaimer_linebreak_issues))
        if nav_bracket_leak_issues: glossary_parts.append("Nav path 표기 변환 점검:\n" + "\n".join(f"- {msg}" for msg in nav_bracket_leak_issues))
        glossary_section = "\n\n".join(glossary_parts) if glossary_parts else "별도 지적 사항 없음."

        # RAG consistency check (Post-translation/audit, does not use LLM)
        rag_text = "별도 설정 없음."
        rag_json = "[]"
        
        # Skip RAG for placeholder or very short/empty text
        if is_placeholder and not pre_mismatch:
            rag_text = "[건너뜀: 짧은/무의미]"
        else:
            results, err = await self._rag_lookup(source, sheet_name, source_lang, rag_identity_match)
            if err:
                rag_text = f"RAG 조회 오류: {err}"
                rag_json = json.dumps([{"error": err}], ensure_ascii=False)
            elif results:
                # 1. Human-readable text
                text_lines = []
                for i, r in enumerate(results, 1):
                    match_info = f"{r['match_type'].upper()}"
                    if r['match_type'] == "semantic":
                        match_info += f" ({r['similarity_score']*100:.1f}%)"

                    text_lines.append(f"[사례 {i}] {match_info} | {r.get('story_id','')} | {r['section_code']}\n- 번역: {r['target']}")
                rag_text = "\n\n".join(text_lines)

                # 2. JSON payload
                rag_data = []
                for r in results:
                    rag_data.append({
                        "type": r['match_type'],
                        "score": round(r['similarity_score'] * 100, 1),
                        "story_id": r.get('story_id', ''),
                        "section": r.get('section_code', ''),
                        "source": r.get('source', ''),
                        "target": r.get('target', '')
                    })
                rag_json = json.dumps(rag_data, ensure_ascii=False)
            elif getattr(self, "rag_retriever", None) and self.rag_retriever.is_available():
                rag_text = "유사 과거 사례 없음."
                rag_json = "[]"

        # Result container
        res = {
            "sheet_name": sheet_name,
            "cell_ref": cell_ref,
            "source": source,
            "target": target,
            "case_section": case_section,
            "glossary_section": glossary_section,
            "rag_text": rag_text,
            "rag_json": rag_json,
            "back_translation": "",
            "ai_text": "",
            "ai_json": "",
            "hard_constraint_card": constraint_card,
            "constraint_validation": constraint_validation,
        }

        # Skip Logic
        if is_placeholder and not pre_mismatch:
            res["back_translation"] = "[건너뜀: 짧은/무의미]"
            res["ai_text"] = "[건너뜀: 짧은/무의미]"
            res["ai_json"] = "[]"
            return res

        # LLM Logic
        if skip_llm:
            res["back_translation"] = "[사전 감지로 LLM 호출 생략]"
            res["ai_text"] = "※ 용어집 사전 감지 결과를 우선 검토하세요."
            res["ai_json"] = "[]"
        else:
            # LLM Calls
            qa_task = self._with_semaphore(
                self.check_with_llm_qa(source, target, source_lang, tgt_lang, tgt_code, row_key=row_key,
                                       constraint_card=constraint_card,
                                       pending_activation_candidates=pending_activation_candidates)
            )
            
            if self.no_backtranslation:
                ai_tuple = await qa_task
                res["ai_text"], res["ai_json"] = ai_tuple
                res["back_translation"] = "[역번역 비활성화됨]"
            else:
                bt_ref_lang = self.backtranslation_lang or source_lang
                bt_task = self._with_semaphore(
                    self.get_back_translation(target, tgt_lang, bt_ref_lang)
                )
                ai_tuple, bt = await asyncio.gather(qa_task, bt_task)
                res["ai_text"], res["ai_json"] = ai_tuple
                res["back_translation"] = bt
        
        return res

    # ----------------- Generator Method for SSE -----------------
    async def run_inspection_async_generator(
        self,
        source_file_path,
        target_file_path,
        cell_range,
        source_lang,
        target_lang,
        target_lang_code,
        sheet_lang_map,
        glossary_url=None,
        selected_sheets: list = None,
        glossary_file_path: str = None,
        source_sheet_name: str = None,
        rag_identity_match: bool = True,
        source_groups: list = None
    ):
        """
        Yields events:
        {"type": "log", "message": "..."}
        {"type": "progress", "current": n, "total": m, "percent": ...}
        {"type": "result_chunk", "data": formatted_string}
        {"type": "complete", "total": m, "output_data": all_text}
        """
        def _fmt_result(res):
            return render_finding(res)

        match = re.search(r"(?:story[_ -]?)?(\d{3})(?:\D|$)", os.path.basename(str(source_file_path)), re.IGNORECASE)
        self._inspection_story_id = match.group(1) if match else None

        # --- Multi-Source Mode ---
        if source_groups:
            all_group_results = []
            grand_total = 0

            for g_idx, group in enumerate(source_groups):
                g_src_sheet = group.get("source_sheet")
                g_tgt_sheets = group.get("target_sheets", [])
                g_src_info = sheet_lang_map.get(g_src_sheet, {})
                g_src_lang = g_src_info.get("lang", source_lang)
                g_src_code = g_src_info.get("code", g_src_lang)
                label = f"[그룹 {g_idx + 1}: {g_src_sheet}]"

                # Load glossary for this group's source
                if glossary_file_path:
                    yield {"type": "log", "message": f"{label} 용어집 로드 중 (기준: {g_src_code})..."}
                    msg = await self.load_glossary_from_file(glossary_file_path, g_src_code)
                    yield {"type": "log", "message": msg}

                # Load Excel data for this group's target sheets
                yield {"type": "log", "message": f"{label} 시트 데이터 추출 중..."}
                try:
                    g_logs = []
                    g_data, g_sheets = self.load_excel_data(
                        source_file_path, target_file_path,
                        cell_range=cell_range,
                        selected_sheets=g_tgt_sheets if g_tgt_sheets else None,
                        log_func=lambda m: g_logs.append(m),
                        source_sheet_name=g_src_sheet
                    )
                    for m in g_logs:
                        yield {"type": "log", "message": m}
                except Exception as e:
                    yield {"type": "error", "message": f"{label} 데이터 로드 오류: {e}"}
                    continue

                g_total = len(g_data)
                grand_total += g_total
                yield {"type": "log", "message": f"{label} 검수 항목: {g_total}개"}
                yield {"type": "progress", "current": 0, "total": grand_total, "percent": 0}

                if g_total == 0:
                    continue

                async def _process(index, item, src_lang=g_src_lang):
                    res = await self.process_item(item, src_lang, target_lang, sheet_lang_map, target_lang_code, rag_identity_match=rag_identity_match)
                    return index, res

                g_ordered = [None] * g_total
                g_done = 0
                g_tasks = [_process(i, item) for i, item in enumerate(g_data)]

                for future in asyncio.as_completed(g_tasks):
                    try:
                        idx, res = await future
                        g_done += 1
                        g_ordered[idx] = _fmt_result(res)
                        yield {
                            "type": "progress",
                            "current": len(all_group_results) + g_done,
                            "total": grand_total,
                            "percent": int((len(all_group_results) + g_done) / grand_total * 100) if grand_total else 0,
                            "log": f"{label} [{res['sheet_name']}] {res['cell_ref']} 검수 완료"
                        }
                    except Exception as e:
                        yield {"type": "log", "message": f"{label} 항목 오류: {e}"}

                all_group_results.extend([r for r in g_ordered if r is not None])

            output_data = self._render_report(
                title="번역 검수 보고서",
                source_file_id=source_file_path,
                findings=all_group_results,
                summary_lines=[
                    f"총 검수 항목: {grand_total}개 ({len(source_groups)}개 소스 그룹)",
                    f"용어집 항목: {len(self.glossary)}",
                ],
            )
            yield {"type": "complete", "total": grand_total, "output_data": output_data}
            return
        # --- End Multi-Source Mode ---

        yield {"type": "log", "message": "용어집 로드 시작..."}

        # If source_sheet_name is provided, use it to infer source_lang
        if source_sheet_name and sheet_lang_map:
            source_info = sheet_lang_map.get(source_sheet_name)
            if source_info:
                source_lang = source_info.get('lang', source_lang)
        
        # Load Glossary
        if glossary_file_path:
             src_lookup = source_lang
             if source_sheet_name and sheet_lang_map:
                 s_info = sheet_lang_map.get(source_sheet_name)
                 if s_info:
                     src_lookup = s_info.get('code', s_info.get('lang', source_lang))
                 else:
                     # Sheet not in map: infer source language from sheet name pattern
                     sn = source_sheet_name.lower()
                     if any(x in sn for x in ["kr", "한국", "korean"]):
                         src_lookup = "Korean"
                     elif any(x in sn for x in ["us", "en_us", "미국"]):
                         src_lookup = "English"

             yield {"type": "log", "message": f"용어집 파일 로드 중: {os.path.basename(glossary_file_path)} (기준: {src_lookup})"}
             msg = await self.load_glossary_from_file(glossary_file_path, src_lookup)
             yield {"type": "log", "message": msg}
        
        yield {"type": "log", "message": "엑셀 파일 및 시트 분석 중..."}
        try:
            # Use explicit selected_sheets if provided, otherwise fallback to map keys
            sel_sheets = selected_sheets if selected_sheets else (list(sheet_lang_map.keys()) if sheet_lang_map else None)
            
            log_messages = []
            def collect_log(m): log_messages.append(m)

            all_data, processed_sheets = self.load_excel_data(
                source_file_path, target_file_path, cell_range=cell_range, selected_sheets=sel_sheets, log_func=collect_log, source_sheet_name=source_sheet_name
            )
            for m in log_messages:
                yield {"type": "log", "message": m}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        total_items = len(all_data)
        yield {"type": "log", "message": f"검수 대상: {len(processed_sheets)}개 시트, 총 {total_items}개 항목"}
        yield {"type": "progress", "current": 0, "total": total_items, "percent": 0}

        if total_items == 0:
            yield {"type": "complete", "total": 0, "output_data": self._render_report(
                title="번역 검수 보고서",
                source_file_id=source_file_path,
                findings=[],
                summary_lines=["검수할 데이터가 없습니다."],
            )}
            return

        # Wrapper to track index
        async def process_with_index(index, item):
            res = await self.process_item(item, source_lang, target_lang, sheet_lang_map, target_lang_code, rag_identity_match=rag_identity_match)
            return index, res

        # Create tasks with index
        tasks = [
            process_with_index(i, item)
            for i, item in enumerate(all_data)
        ]

        completed_count = 0
        # Initialize list to store results in order
        ordered_results = [None] * total_items
        
        # Use asyncio.as_completed to yield progress, but store by index
        for future in asyncio.as_completed(tasks):
            try:
                index, res = await future
                completed_count += 1
                
                # Format result
                fmt_result = _fmt_result(res)
                
                # Store in the correct slot
                ordered_results[index] = fmt_result
                
                percent = int((completed_count / total_items) * 100)
                yield {
                    "type": "progress",
                    "current": completed_count,
                    "total": total_items,
                    "percent": percent,
                    "log": f"[{res['sheet_name']}] {res['cell_ref']} 검수 완료"
                }
            except Exception as e:
                yield {"type": "log", "message": f"항목 처리 중 오류 발생: {str(e)}"}
                continue

        # Final assembly using ORDERED results (None = the item failed)
        full_output = self._render_report(
            title="번역 검수 보고서",
            source_file_id=source_file_path,
            findings=[r for r in ordered_results if r is not None],
            summary_lines=[
                f"총 검수 항목: {total_items}개",
                f"용어집 항목: {len(self.glossary)}",
            ],
        )
        yield {"type": "complete", "total": total_items, "output_data": full_output}

    # ----------------- Integrated Translation & Audit Generator -----------------
    async def run_integrated_pipeline_generator(
        self,
        source_file_path,
        cell_range,
        bx_style_on,
        sheet_lang_map,
        translation_model="gemini-3.6-flash",
        audit_model="gpt-5.2",
        translation_thinking_budget: int | None = None,
        glossary_file_path=None,
        selected_sheets=None,
        source_sheet_name=None,
        skip_audit=False,
        source_lang="English",
        rag_identity_match=True,
        source_groups=None
    ):
        """
        Translates source text, audits it, and saves it to a NEW Excel file.
        Yields SSE events same as inspection.
        """
        yield {"type": "log", "message": "Starting Integrated Translation & Audit Pipeline..."}

        # Same story-id convention as run_inspection_async_generator, so an
        # occurrence-activation-manifest entry (keyed by story/cell/term) is
        # actually reachable here too, not just from the read-only audit flow.
        match = re.search(r"(?:story[_ -]?)?(\d{3})(?:\D|$)", os.path.basename(str(source_file_path)), re.IGNORECASE)
        story_for_run = match.group(1) if match else None

        # --- Multi-Source Mode ---
        if source_groups:
            try:
                wb = openpyxl.load_workbook(source_file_path, rich_text=True)
            except Exception as e:
                yield {"type": "error", "message": f"Excel Load Error: {str(e)}"}
                return

            all_fmt_results = []
            expected_texts: dict[tuple[str, str], str] = {}
            grand_total = 0
            completed_so_far = 0

            for g_idx, group in enumerate(source_groups):
                g_src_sheet = group.get("source_sheet")
                g_tgt_sheets = group.get("target_sheets", [])
                label = f"[그룹 {g_idx + 1}: {g_src_sheet}]"

                if not g_src_sheet or g_src_sheet not in wb.sheetnames:
                    yield {"type": "log", "message": f"{label} 소스 시트 없음, 건너뜀"}
                    continue

                g_src_info = sheet_lang_map.get(g_src_sheet, {})
                g_src_lang = g_src_info.get("lang", source_lang)
                g_src_code = g_src_info.get("code", g_src_lang)

                if glossary_file_path:
                    yield {"type": "log", "message": f"{label} 용어집 로드 중 (기준: {g_src_code})..."}
                    msg = await self.load_glossary_from_file(glossary_file_path, g_src_code)
                    yield {"type": "log", "message": msg}

                source_ws = wb[g_src_sheet]
                source_data = []
                extracted_coords = set()
                for current_range in [r.strip() for r in cell_range.split(',') if r.strip()]:
                    try:
                        rows = source_ws[current_range]
                        if not isinstance(rows, (tuple, list)):
                            rows = ((rows,),)
                        for row in rows:
                            for cell in row:
                                if cell.coordinate in extracted_coords:
                                    continue
                                extracted_coords.add(cell.coordinate)
                                val = str(cell.value).strip() if cell.value is not None else ""
                                if val and val.lower() != "x":
                                    row_key = self._get_row_key(source_ws, cell.row)
                                    source_data.append({'text': str(cell.value).strip(), 'coord': cell.coordinate, 'row_key': row_key})
                    except Exception as e:
                        yield {"type": "log", "message": f"{label} 범위 오류 {current_range}: {e}"}

                if not source_data:
                    yield {"type": "log", "message": f"{label} 소스 텍스트 없음, 건너뜀"}
                    continue

                g_resolved_tgt = [s for s in g_tgt_sheets if s in wb.sheetnames and s != g_src_sheet and s in sheet_lang_map]
                if not g_resolved_tgt:
                    yield {"type": "log", "message": f"{label} 유효한 타겟 시트 없음, 건너뜀"}
                    continue

                g_total = len(source_data) * len(g_resolved_tgt)
                grand_total += g_total
                yield {"type": "log", "message": f"{label} 시트 {len(g_resolved_tgt)}개, {g_total}개 셀 번역 예정"}
                yield {"type": "progress", "current": completed_so_far, "total": grand_total, "percent": int(completed_so_far / grand_total * 100) if grand_total else 0}

                async def _cell_worker_multi(ws, index, item, tgt_lang, tgt_lang_code, captured_src_lang=g_src_lang):
                    source_text = item['text']
                    coord = item['coord']
                    row_key = item.get("row_key", "")

                    glossary_dict = self._get_glossary_context_as_dict(tgt_lang_code, source_text=source_text, skip_deactivated=True, row_key=row_key)
                    if not glossary_dict:
                        glossary_dict = None

                    rag_context_str, logs = await self._fetch_rag_prompt_context(
                        source_text, ws.title, captured_src_lang, coord, rag_identity_match=rag_identity_match
                    )

                    translation = await self._run_llm_translation(
                        source_text, tgt_lang, model_name=translation_model, bx_style_on=bx_style_on,
                        glossary_context=glossary_dict, rag_context=rag_context_str, row_key=row_key,
                        source_lang=captured_src_lang, rag_identity_match=rag_identity_match, target_lang_code=tgt_lang_code,
                        thinking_budget=translation_thinking_budget,
                        constraint_card=self.resolve_constraints(source_text, tgt_lang_code, row_key=row_key, story=story_for_run, cell=coord),
                    )

                    original_target_terms = []
                    for s_term in (self._get_relevant_glossary_terms(source_text) or []):
                        meta = self.glossary.get(s_term)
                        if meta:
                            if not self.is_term_active_for_occurrence(s_term, meta.get("rule", ""), story=story_for_run, cell=coord):
                                continue
                            target_val = self._get_target_val(meta["targets"], tgt_lang_code)
                            if target_val:
                                clean_val = re.sub(r'\(.*?\)', '', target_val).strip()
                                for extract_str in (clean_val, target_val):
                                    if not extract_str:
                                        continue
                                    original_target_terms.append(extract_str.strip())
                                    split_terms = [x.strip() for x in re.split(r'[,/]', extract_str) if x.strip()]
                                    if len(split_terms) > 1:
                                        original_target_terms.extend(split_terms)

                    target_cell = ws[coord]
                    target_cell.value = self._apply_rich_text(translation, original_target_terms, base_font=target_cell.font)

                    item_data = {"cell_ref": coord, "sheet_name": ws.title, "source": source_text, "target": translation, "row_key": row_key, "story_id": story_for_run}
                    if skip_audit:
                        res = {**item_data, "case_section": "[Bypassed]", "glossary_section": "[Bypassed]", "rag_text": "[Bypassed]", "rag_json": "[]", "back_translation": "[Bypassed]", "ai_text": "[Bypassed: Translate Only Mode]", "ai_json": "{}", "logs": logs}
                    else:
                        res = await self.process_item(item_data, captured_src_lang, tgt_lang, sheet_lang_map, tgt_lang_code, rag_identity_match=rag_identity_match)
                        res["logs"] = logs
                        self._write_backtranslation_cell(ws, coord, res.get("back_translation"))

                    return index, res

                g_tasks = []
                local_idx = 0
                for sheet_name in g_resolved_tgt:
                    ws = wb[sheet_name]
                    sheet_info = sheet_lang_map[sheet_name]
                    for item in source_data:
                        g_tasks.append(_cell_worker_multi(ws, local_idx, item, sheet_info['lang'], sheet_info['code']))
                        local_idx += 1

                g_ordered = [None] * g_total
                g_done = 0

                for future in asyncio.as_completed(g_tasks):
                    try:
                        local_idx_res, res = await future
                        g_done += 1
                        completed_so_far += 1
                        for msg in res.get("logs", []):
                            yield {"type": "log", "message": msg}
                        fmt = render_finding(res, label=label)
                        g_ordered[local_idx_res] = fmt
                        expected_texts[(res['sheet_name'], res['cell_ref'])] = res['target']
                        yield {
                            "type": "progress",
                            "current": completed_so_far,
                            "total": grand_total,
                            "percent": int(completed_so_far / grand_total * 100) if grand_total else 0,
                            "log": f"{label} [{res['sheet_name']}] {res['cell_ref']} 처리 완료"
                        }
                    except Exception as e:
                        yield {"type": "log", "message": f"{label} 셀 처리 오류: {str(e)}"}

                all_fmt_results.extend([r for r in g_ordered if r is not None])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_excel_path = source_file_path.replace(".xlsx", f"_translated_{timestamp}.xlsx")
            self._harden_workbook_whitespace_runs(wb)
            wb.save(out_excel_path)
            self._verify_saved_cell_text(out_excel_path, expected_texts)
            output_data = self._render_report(
                title="번역 통합 검수 보고서",
                source_file_id=source_file_path,
                findings=all_fmt_results,
                summary_lines=[
                    f"총 항목: {completed_so_far}개 ({len(source_groups)}개 소스 그룹)",
                    f"번역본: {os.path.basename(out_excel_path)}",
                ],
                workflow="combined_review",
                translation_model=translation_model,
            )
            yield {"type": "complete", "output_data": output_data, "excel_path": out_excel_path}
            return
        # --- End Multi-Source Mode ---

        # 1. Load Excel
        try:
            wb = openpyxl.load_workbook(source_file_path, rich_text=True)
        except Exception as e:
            yield {"type": "error", "message": f"Excel Load Error: {str(e)}"}
            return

        if not source_sheet_name:
            source_sheet_name = "KR(한국)"
        if source_sheet_name not in wb.sheetnames:
            yield {"type": "error", "message": f"Source sheet '{source_sheet_name}' not found."}
            return

        # Infer source lang
        source_lang = "Korean"
        if source_sheet_name in sheet_lang_map:
            source_lang = sheet_lang_map[source_sheet_name].get("lang", source_lang)

        # 2. Load Glossary
        if glossary_file_path:
            src_lookup = source_lang
            if source_sheet_name in sheet_lang_map:
                s_info = sheet_lang_map[source_sheet_name]
                src_lookup = s_info.get('code', s_info.get('lang', source_lang))
            else:
                # Sheet not in map: infer source language from sheet name pattern
                sn = (source_sheet_name or "").lower()
                if any(x in sn for x in ["kr", "한국", "korean"]):
                    src_lookup = "Korean"
                elif any(x in sn for x in ["us", "en_us", "미국"]):
                    src_lookup = "English"

            yield {"type": "log", "message": f"Loading glossary (Match Base: {src_lookup})..."}
            msg = await self.load_glossary_from_file(glossary_file_path, src_lookup)
            yield {"type": "log", "message": msg}

        source_ws = wb[source_sheet_name]
        
        # 3. Extract source data
        source_data = []
        extracted_coords = set()
        # 콤마로 구분된 여러 범위나 단일 셀(c16, c21)을 모두 지원합니다.
        range_parts = [r.strip() for r in cell_range.split(',') if r.strip()]
        
        for current_range in range_parts:
            try:
                rows = source_ws[current_range]
                if not isinstance(rows, tuple) and not isinstance(rows, list):
                    rows = ((rows,),)
                for row in rows:
                    for cell in row:
                        if cell.coordinate in extracted_coords:
                            continue
                        extracted_coords.add(cell.coordinate)
                        
                        val = str(cell.value).strip() if cell.value is not None else ""
                        if val and val.lower() != "x":
                            row_idx = cell.row
                            row_key = self._get_row_key(source_ws, row_idx)
                            source_data.append({'text': str(cell.value).strip(), 'coord': cell.coordinate, 'row_key': row_key})
            except Exception as e:
                yield {"type": "error", "message": f"Error accessing range {current_range}: {e}"}
                continue

        if not source_data:
            yield {"type": "error", "message": "No source text found in range."}
            return

        # 4. Identify target sheets
        available_sheets = wb.sheetnames
        if selected_sheets:
            target_sheets = [s for s in selected_sheets if s in available_sheets and s != source_sheet_name]
        else:
            target_sheets = [s for s in available_sheets if s in sheet_lang_map and s != source_sheet_name]

        total_cells = len(source_data) * len(target_sheets)
        yield {"type": "log", "message": f"Plan: {len(target_sheets)} sheets, Total {total_cells} cells."}
        yield {"type": "progress", "current": 0, "total": total_cells, "percent": 0}

        # 5. Process in parallel (cells)
        completed_count = 0
        
        async def cell_worker(ws, index, item, target_lang, target_lang_code):
            nonlocal completed_count
            source_text = item['text']
            coord = item['coord']
            row_key = item.get("row_key", "")
            
            # Step 1: Translate
            # Translating: filter out terms explicitly deactivated to save tokens and not force LLM behavior
            glossary_dict = self._get_glossary_context_as_dict(
                target_lang_code,
                source_text=source_text,
                skip_deactivated=True,
                row_key=row_key,
            )
            if not glossary_dict:
                glossary_dict = None

            # RAG Injection
            rag_context_str, logs = await self._fetch_rag_prompt_context(
                source_text, ws.title, source_lang, coord, rag_identity_match=rag_identity_match
            )

            translation = await self._run_llm_translation(
                source_text,
                target_lang,
                model_name=translation_model,
                bx_style_on=bx_style_on,
                glossary_context=glossary_dict,
                rag_context=rag_context_str,
                row_key=row_key,
                source_lang=source_lang,
                rag_identity_match=rag_identity_match,
                target_lang_code=target_lang_code,
                thinking_budget=translation_thinking_budget,
                constraint_card=self.resolve_constraints(source_text, target_lang_code, row_key=row_key, story=story_for_run, cell=coord),
            )

            # Extract plain glossary targets (removing EXCEPTION strings if any)
            # Use case-insensitive target language code matching
            original_target_terms = []

            relevant_terms_for_highlight = self._get_relevant_glossary_terms(source_text)
            if relevant_terms_for_highlight:
                for s_term in relevant_terms_for_highlight:
                    meta = self.glossary.get(s_term)
                    if meta:
                        # Skip terms inactive for this story/cell occurrence (manifest-aware)
                        if not self.is_term_active_for_occurrence(s_term, meta.get("rule", ""), story=story_for_run, cell=coord):
                            continue

                        t_meta = meta["targets"]
                        target_val = self._get_target_val(t_meta, target_lang_code)

                        if target_val:
                            clean_val = re.sub(r'\(.*?\)', '', target_val).strip()
                            
                            # 반점(,)이나 슬래시(/)로 여러 단어가 기재된 경우(예: "단어1, 단어2") 각각 하이라이트 되도록 분리
                            for extract_str in (clean_val, target_val):
                                if not extract_str: continue
                                original_target_terms.append(extract_str.strip())
                                # 정규식으로 분리 (콤마 또는 슬래시 기준, 양옆 공백까지 같이 잡아냄)
                                split_terms = [x.strip() for x in re.split(r'[,/]', extract_str) if x.strip()]
                                if len(split_terms) > 1:
                                    original_target_terms.extend(split_terms)

            target_cell = ws[coord]
            target_cell.value = self._apply_rich_text(translation, original_target_terms, base_font=target_cell.font)

            item_data = {
                "cell_ref": coord,
                "sheet_name": ws.title,
                "source": source_text,
                "target": translation,
                "row_key": row_key,
                "story_id": story_for_run,
            }

            if skip_audit:
                res = {
                    "sheet_name": ws.title,
                    "cell_ref": coord,
                    "source": source_text,
                    "target": translation,
                    "case_section": "[Bypassed]",
                    "glossary_section": "[Bypassed]",
                    "rag_text": "[Bypassed]",
                    "rag_json": "[]",
                    "back_translation": "[Bypassed]",
                    "ai_text": "[Bypassed: Translate Only Mode]",
                    "ai_json": "{}",
                    "logs": logs
                }
            else:
                res = await self.process_item(item_data, source_lang, target_lang, sheet_lang_map, target_lang_code, rag_identity_match=rag_identity_match)
                res["logs"] = logs
                self._write_backtranslation_cell(ws, coord, res.get("back_translation"))

            return index, res

        # Flatten all cells into a list of tasks
        tasks = []
        idx = 0
        for sheet_name in target_sheets:
            ws = wb[sheet_name]
            sheet_info = sheet_lang_map[sheet_name]
            target_lang = sheet_info['lang']
            target_lang_code = sheet_info['code']
            
            for item in source_data:
                tasks.append(cell_worker(ws, idx, item, target_lang, target_lang_code))
                idx += 1

        ordered_results = [None] * len(tasks)
        expected_texts: dict[tuple[str, str], str] = {}
        
        # Yield per-cell progress
        for future in asyncio.as_completed(tasks):
            try:
                index, res = await future
                completed_count += 1
                
                # yield UI logs from cell_worker
                for msg in res.get("logs", []):
                    yield {"type": "log", "message": msg}

                ordered_results[index] = render_finding(res)
                expected_texts[(res['sheet_name'], res['cell_ref'])] = res['target']
                
                percent = int((completed_count / total_cells) * 100)
                yield {
                    "type": "progress", 
                    "current": completed_count, 
                    "total": total_cells, 
                    "percent": percent, 
                    "log": f"[{res['sheet_name']}] {res['cell_ref']} 처리 완료"
                }
            except Exception as e:
                yield {"type": "log", "message": f"Cell processing error: {str(e)}"}

        # 6. Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_excel_path = source_file_path.replace(".xlsx", f"_translated_{timestamp}.xlsx")
        self._harden_workbook_whitespace_runs(wb)
        wb.save(out_excel_path)
        self._verify_saved_cell_text(out_excel_path, expected_texts)
        
        report_text = self._render_report(
            title="번역 통합 검수 보고서",
            source_file_id=source_file_path,
            findings=[r for r in ordered_results if r is not None],
            summary_lines=[
                f"총 항목: {completed_count}개",
                f"번역본: {os.path.basename(out_excel_path)}",
            ],
            workflow="combined_review",
            translation_model=translation_model,
        )
        
        yield {
            "type": "complete", 
            "output_data": report_text, 
            "excel_path": out_excel_path
        }


    async def run_highlight_only_pipeline_generator(
        self,
        source_file_path,
        cell_range,
        sheet_lang_map,
        glossary_file_path=None,
        selected_sheets=None,
        source_sheet_name=None,
        source_lang="English",
        source_groups=None,
        include_source_sheets=False
    ):
        """
        Highlights glossary terms in target sheets without translating.
        Relies on the target text already being present in the Excel file.
        """
        yield {"type": "log", "message": "Starting Highlight Only Pipeline..."}

        # Same story-id convention as the inspection/translate pipelines, so an
        # occurrence-activation-manifest entry (story/cell/term) is reachable
        # here too instead of falling back to the glossary's global rule only.
        match = re.search(r"(?:story[_ -]?)?(\d{3})(?:\D|$)", os.path.basename(str(source_file_path)), re.IGNORECASE)
        story_for_run = match.group(1) if match else None

        # --- Multi-Source Mode ---
        if source_groups:
            try:
                wb = openpyxl.load_workbook(source_file_path, rich_text=True)
            except Exception as e:
                yield {"type": "error", "message": f"Excel Load Error: {str(e)}"}
                return

            grand_total = 0
            completed_so_far = 0
            all_highlight_stats = {}
            all_detail_lines = []

            for g_idx, group in enumerate(source_groups):
                g_src_sheet = group.get("source_sheet")
                g_tgt_sheets = group.get("target_sheets", [])
                label = f"[그룹 {g_idx + 1}: {g_src_sheet}]"

                if not g_src_sheet or g_src_sheet not in wb.sheetnames:
                    yield {"type": "log", "message": f"{label} 소스 시트 없음, 건너뜀"}
                    continue

                g_src_info = sheet_lang_map.get(g_src_sheet, {})
                g_src_lang = g_src_info.get("lang", source_lang)
                g_src_code = g_src_info.get("code", g_src_lang)

                if glossary_file_path:
                    yield {"type": "log", "message": f"{label} 용어집 로드 중 (기준: {g_src_code})..."}
                    msg = await self.load_glossary_from_file(glossary_file_path, g_src_code)
                    yield {"type": "log", "message": msg}

                if not self.glossary:
                    yield {"type": "log", "message": f"{label} 용어집 없음, 건너뜀"}
                    continue

                source_ws = wb[g_src_sheet]
                source_data = []
                extracted_coords = set()
                for current_range in [r.strip() for r in cell_range.split(',') if r.strip()]:
                    try:
                        rows = source_ws[current_range]
                        if not isinstance(rows, (tuple, list)):
                            rows = ((rows,),)
                        for row in rows:
                            for cell in row:
                                if cell.coordinate in extracted_coords:
                                    continue
                                extracted_coords.add(cell.coordinate)
                                val = str(cell.value).strip() if cell.value is not None else ""
                                if val and val.lower() != "x":
                                    row_key = self._get_row_key(source_ws, cell.row)
                                    source_data.append({'text': val, 'coord': cell.coordinate, 'row_key': row_key})
                    except Exception as e:
                        yield {"type": "log", "message": f"{label} 범위 오류 {current_range}: {e}"}

                if not source_data:
                    yield {"type": "log", "message": f"{label} 소스 텍스트 없음, 건너뜀"}
                    continue

                g_resolved_tgt = [
                    s for s in g_tgt_sheets
                    if s in wb.sheetnames
                    and (include_source_sheets or s != g_src_sheet)
                    and s in sheet_lang_map
                ]
                if not g_resolved_tgt:
                    yield {"type": "log", "message": f"{label} 유효한 타겟 시트 없음, 건너뜀"}
                    continue

                g_total = len(source_data) * len(g_resolved_tgt)
                grand_total += g_total
                yield {"type": "log", "message": f"{label} 시트 {len(g_resolved_tgt)}개, {g_total}개 셀 하이라이트 예정"}
                yield {"type": "progress", "current": completed_so_far, "total": grand_total, "percent": int(completed_so_far / grand_total * 100) if grand_total else 0}

                for sheet_name in g_resolved_tgt:
                    all_highlight_stats[f"{label} {sheet_name}"] = {"cells_processed": 0, "total_highlights": 0}

                async def _highlight_worker_multi(ws, index, item, tgt_lang, tgt_lang_code):
                    source_text = item['text']
                    coord = item['coord']
                    row_key = item.get("row_key", "")
                    target_cell = ws[coord]
                    # Rich-text highlighting must not normalize the cell value.  In particular,
                    # ``strip()`` here silently deleted leading/trailing spaces on save.
                    target_text = self._cell_text_for_highlighting(target_cell.value)
                    logs = []
                    highlight_count = 0

                    if target_text and target_text.lower() != "x":
                        for issue in self._check_glossary_brackets(source_text, target_text, tgt_lang_code, tgt_lang, row_key=row_key):
                            logs.append(issue)
                        for issue in self._precheck_glossary_mismatch(source_text, target_text, tgt_lang_code):
                            logs.append(issue)
                        for issue in self._check_glossary_casing(source_text, target_text, tgt_lang_code):
                            logs.append(issue)
                        for issue in self._check_brand_concatenation(source_text, target_text, tgt_lang_code, tgt_lang):
                            logs.append(issue)
                        for issue in self._check_disclaimer_linebreak(source_text, target_text, row_key):
                            logs.append(issue)
                        for issue in self._check_nav_path_bracket_leak(target_text, row_key):
                            logs.append(issue)

                        original_target_terms = []
                        for s_term in (self._get_relevant_glossary_terms(source_text) or []):
                            meta = self.glossary.get(s_term)
                            if meta:
                                if not self.is_term_active_for_occurrence(s_term, meta.get("rule", ""), story=story_for_run, cell=coord):
                                    continue
                                target_val = self._get_target_val(meta["targets"], tgt_lang_code)
                                if target_val:
                                    clean_val = re.sub(r'\(.*?\)', '', target_val).strip()
                                    for extract_str in (clean_val, target_val):
                                        if not extract_str:
                                            continue
                                        original_target_terms.append(extract_str.strip())
                                        split_terms = [x.strip() for x in re.split(r'[,/]', extract_str) if x.strip()]
                                        if len(split_terms) > 1:
                                            original_target_terms.extend(split_terms)

                        if original_target_terms:
                            target_cell.value = self._apply_rich_text(target_text, original_target_terms, base_font=target_cell.font)
                            sorted_kw = sorted([k.strip() for k in original_target_terms if k.strip()], key=len, reverse=True)
                            if sorted_kw:
                                pattern = '|'.join(re.escape(k) for k in sorted_kw)
                                matches = list(re.finditer(pattern, target_text, flags=re.IGNORECASE))
                                highlight_count = len(matches)
                                if matches:
                                    logs.append(f"[{ws.title}] 셀 {coord}: {highlight_count}개 키워드 하이라이트 적용")

                    return index, {"sheet_name": ws.title, "cell_ref": coord, "highlight_count": highlight_count, "logs": logs}

                g_tasks = []
                local_idx = 0
                for sheet_name in g_resolved_tgt:
                    ws = wb[sheet_name]
                    sheet_info = sheet_lang_map[sheet_name]
                    for item in source_data:
                        g_tasks.append(_highlight_worker_multi(ws, local_idx, item, sheet_info['lang'], sheet_info['code']))
                        local_idx += 1

                g_ordered = [None] * g_total

                for future in asyncio.as_completed(g_tasks):
                    try:
                        local_idx_res, res = await future
                        completed_so_far += 1
                        g_ordered[local_idx_res] = res
                        stat_key = f"{label} {res['sheet_name']}"
                        if stat_key in all_highlight_stats:
                            all_highlight_stats[stat_key]["cells_processed"] += 1
                            all_highlight_stats[stat_key]["total_highlights"] += res["highlight_count"]
                        for msg in res.get("logs", []):
                            yield {"type": "log", "message": msg}
                        yield {
                            "type": "progress",
                            "current": completed_so_far,
                            "total": grand_total,
                            "percent": int(completed_so_far / grand_total * 100) if grand_total else 0,
                            "log": f"{label} [{res['sheet_name']}] {res['cell_ref']} 검토 완료"
                        }
                    except Exception as e:
                        yield {"type": "log", "message": f"{label} 셀 처리 오류: {str(e)}"}

                for res in g_ordered:
                    if res and res.get("logs"):
                        issue_logs = [log for log in res["logs"] if any(k in log for k in ["[오류", "[미적용", "[대소문자", "[공백"])]
                        if issue_logs:
                            all_detail_lines.append(f"\n{label} [{res['sheet_name']} 시트 | {res['cell_ref']} 셀]")
                            for log in issue_logs:
                                all_detail_lines.append(f"  - {log}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_excel_path = source_file_path.replace(".xlsx", f"_highlighted_{timestamp}.xlsx")
            self._harden_workbook_whitespace_runs(wb)
            wb.save(out_excel_path)

            report_text = self._render_highlight_report(
                source_file_id=source_file_path,
                out_excel_path=out_excel_path,
                summary_lines=[
                    f"총 대상 셀 수: {grand_total}개 ({len(source_groups)}개 소스 그룹)",
                    *[
                        f"{k} | 처리 완료 셀: {v['cells_processed']} | 총 하이라이트: {v['total_highlights']}개"
                        for k, v in all_highlight_stats.items()
                    ],
                ],
                detail_lines=all_detail_lines,
            )
            yield {"type": "complete", "output_data": report_text, "excel_path": out_excel_path}
            return
        # --- End Multi-Source Mode ---

        try:
            wb = openpyxl.load_workbook(source_file_path, rich_text=True)
        except Exception as e:
            yield {"type": "error", "message": f"Excel Load Error: {str(e)}"}
            return

        if not source_sheet_name:
            source_sheet_name = "KR(한국)"
        if source_sheet_name not in wb.sheetnames:
            yield {"type": "error", "message": f"Source sheet '{source_sheet_name}' not found."}
            return

        if source_sheet_name in sheet_lang_map:
            source_lang = sheet_lang_map[source_sheet_name].get("lang", source_lang)

        # Load Glossary
        if glossary_file_path:
            src_lookup = source_lang
            if source_sheet_name in sheet_lang_map:
                s_info = sheet_lang_map[source_sheet_name]
                src_lookup = s_info.get('code', s_info.get('lang', source_lang))

            yield {"type": "log", "message": f"Loading glossary (Match Base: {src_lookup})..."}
            msg = await self.load_glossary_from_file(glossary_file_path, src_lookup)
            yield {"type": "log", "message": msg}
            
        if not self.glossary:
            yield {"type": "error", "message": "No glossary data mapped. Skipping highlight."}
            return

        source_ws = wb[source_sheet_name]
        source_data = []
        extracted_coords = set()
        range_parts = [r.strip() for r in cell_range.split(',') if r.strip()]
        
        for current_range in range_parts:
            try:
                rows = source_ws[current_range]
                if not isinstance(rows, tuple) and not isinstance(rows, list):
                    rows = ((rows,),)
                for row in rows:
                    for cell in row:
                        if cell.coordinate in extracted_coords:
                            continue
                        extracted_coords.add(cell.coordinate)
                        val = str(cell.value).strip() if cell.value is not None else ""
                        if val and val.lower() != "x":
                            row_idx = cell.row
                            row_key = self._get_row_key(source_ws, row_idx)
                            source_data.append({'text': val, 'coord': cell.coordinate, 'row_key': row_key})
            except Exception as e:
                yield {"type": "error", "message": f"Error accessing range {current_range}: {e}"}
                continue

        if not source_data:
            yield {"type": "error", "message": "No source text found in range."}
            return

        available_sheets = wb.sheetnames
        if selected_sheets:
            target_sheets = [
                s for s in selected_sheets
                if s in available_sheets
                and (include_source_sheets or s != source_sheet_name)
            ]
        else:
            target_sheets = [
                s for s in available_sheets
                if s in sheet_lang_map
                and (include_source_sheets or s != source_sheet_name)
            ]

        total_cells = len(source_data) * len(target_sheets)
        yield {"type": "log", "message": f"Plan: {len(target_sheets)} sheets, Total {total_cells} cells."}
        yield {"type": "progress", "current": 0, "total": total_cells, "percent": 0}

        completed_count = 0
        highlight_stats = {} # Count of highlights per sheet

        async def highlight_cell_worker(ws, index, item, target_lang, target_lang_code):
            nonlocal completed_count
            source_text = item['text']
            coord = item['coord']
            row_key = item.get("row_key", "")
            
            target_cell = ws[coord]
            # Preserve every character while converting the cell to rich text.
            target_text = self._cell_text_for_highlighting(target_cell.value)
            logs = []
            highlight_count = 0
            bracket_issues = []

            # If target has text, calculate highlights based on source
            if target_text and target_text.lower() != "x":
                # 1. Bracket Check
                bracket_issues = self._check_glossary_brackets(
                    source_text,
                    target_text,
                    target_lang_code,
                    target_lang,
                    row_key=row_key,
                )
                if bracket_issues:
                    logs.extend(bracket_issues)
                
                # 2. Mismatch Check
                mismatch_issues = self._precheck_glossary_mismatch(source_text, target_text, target_lang_code)
                if mismatch_issues:
                    logs.extend(mismatch_issues)
                
                # 3. Casing Check
                casing_issues = self._check_glossary_casing(source_text, target_text, target_lang_code)
                if casing_issues:
                    logs.extend(casing_issues)

                # 4. Brand spacing check (dropped-space concatenation, e.g. SmartThingsFamily)
                brand_concat_issues = self._check_brand_concatenation(source_text, target_text, target_lang_code, target_lang)
                if brand_concat_issues:
                    logs.extend(brand_concat_issues)

                # 5. Disclaimer multi-line '* ' prefix check
                disclaimer_linebreak_issues = self._check_disclaimer_linebreak(source_text, target_text, row_key)
                if disclaimer_linebreak_issues:
                    logs.extend(disclaimer_linebreak_issues)

                # 6. Nav path bracket-leak check (Korean '[...]' path not converted to target quotes)
                nav_bracket_leak_issues = self._check_nav_path_bracket_leak(target_text, row_key)
                if nav_bracket_leak_issues:
                    logs.extend(nav_bracket_leak_issues)

                original_target_terms = []
                relevant_terms = self._get_relevant_glossary_terms(source_text)
                if relevant_terms:
                    for s_term in relevant_terms:
                        meta = self.glossary.get(s_term)
                        if meta:
                            # Skip terms inactive for this story/cell occurrence (manifest-aware)
                            if not self.is_term_active_for_occurrence(s_term, meta.get("rule", ""), story=story_for_run, cell=coord):
                                continue

                            t_meta = meta["targets"]
                            target_val = self._get_target_val(t_meta, target_lang_code)

                            if target_val:
                                clean_val = re.sub(r'\(.*?\)', '', target_val).strip()
                                for extract_str in (clean_val, target_val):
                                    if not extract_str: continue
                                    original_target_terms.append(extract_str.strip())
                                    split_terms = [x.strip() for x in re.split(r'[,/]', extract_str) if x.strip()]
                                    if len(split_terms) > 1:
                                        original_target_terms.extend(split_terms)
                
                # Apply rich text
                if original_target_terms:
                    # We inject the current target cell text instead of a generated translation
                    target_cell.value = self._apply_rich_text(target_text, original_target_terms, base_font=target_cell.font)
                    
                    # Estimate highlights by checking string occurrences
                    sorted_kw = sorted([k.strip() for k in original_target_terms if k.strip()], key=len, reverse=True)
                    if sorted_kw:
                        pattern = '|'.join(re.escape(k) for k in sorted_kw)
                        matches = list(re.finditer(pattern, target_text, flags=re.IGNORECASE))
                        highlight_count = len(matches)
                        if matches:
                            logs.append(f"[{ws.title}] 셀 {coord}: {highlight_count}개 키워드 하이라이트 적용")

            res = {
                "sheet_name": ws.title,
                "cell_ref": coord,
                "highlight_count": highlight_count,
                "logs": logs
            }
            return index, res

        tasks = []
        idx = 0
        for sheet_name in target_sheets:
            ws = wb[sheet_name]
            sheet_info = sheet_lang_map[sheet_name]
            target_lang = sheet_info['lang']
            target_lang_code = sheet_info['code']
            highlight_stats[sheet_name] = {"cells_processed": 0, "total_highlights": 0}
            
            for item in source_data:
                tasks.append(highlight_cell_worker(ws, idx, item, target_lang, target_lang_code))
                idx += 1

        ordered_results = [None] * len(tasks)

        for future in asyncio.as_completed(tasks):
            try:
                index, res = await future
                completed_count += 1
                ordered_results[index] = res # CRITICAL FIX: Populate result list
                
                s_name = res["sheet_name"]
                highlight_stats[s_name]["cells_processed"] += 1
                highlight_stats[s_name]["total_highlights"] += res["highlight_count"]

                for msg in res.get("logs", []):
                    yield {"type": "log", "message": msg}

                percent = int((completed_count / total_cells) * 100)
                yield {
                    "type": "progress", 
                    "current": completed_count, 
                    "total": total_cells, 
                    "percent": percent, 
                    "log": f"[{res['sheet_name']}] {res['cell_ref']} 검토 완료"
                }
            except Exception as e:
                yield {"type": "log", "message": f"Cell processing error: {str(e)}"}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_excel_path = source_file_path.replace(".xlsx", f"_highlighted_{timestamp}.xlsx")
        self._harden_workbook_whitespace_runs(wb)
        wb.save(out_excel_path)
        
        # Highlight Only 모드에서도 괄호 오류 등을 기록하기 위해 ordered_results 활용
        detail_lines = []
        for res in ordered_results:
            if res and res.get("logs"):
                # logs에 [괄호 오류], [미적용], [대소문자] 등이 있으면 상세 기록에 추가
                issue_logs = [log for log in res["logs"] if any(k in log for k in ["[오류", "[미적용", "[대소문자", "[공백"])]
                if issue_logs:
                    detail_lines.append(f"\n[{res['sheet_name']} 시트 | {res['cell_ref']} 셀]")
                    for log in issue_logs:
                        detail_lines.append(f"  - {log}")

        report_text = self._render_highlight_report(
            source_file_id=source_file_path,
            out_excel_path=out_excel_path,
            summary_lines=[
                f"총 대상 셀 수: {total_cells}개",
                *[
                    f"시트: {s_name} | 처리 완료 셀: {stats['cells_processed']} | 총 적용된 하이라이트: {stats['total_highlights']} 개"
                    for s_name, stats in highlight_stats.items()
                ],
            ],
            detail_lines=detail_lines,
        )

        yield {
            "type": "complete", 
            "output_data": report_text, 
            "excel_path": out_excel_path
        }
