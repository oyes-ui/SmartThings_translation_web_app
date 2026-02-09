# -*- coding: utf-8 -*-
"""
backend/checker_service.py
Refactored for Web API usage (FastAPI + SSE)
"""

import openpyxl
import os
from google import genai
from google.genai import types
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import json
import re
import pandas as pd
import io
import urllib.parse
from model_handler import ModelHandler
from bx_guideline_engine import BXGuidelineEngine

# API Keys from Environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------- Case Hard-rule Applicable Languages -----------------
CASE_APPLICABLE_LANG_PREFIXES = {
    "English", "German", "French", "Spanish", "Portuguese",
    "Italian", "Dutch", "Swedish", "Polish", "Turkish",
    "Indonesian", "Vietnamese", "Russian"
}

def _is_case_sensitive_language(lang_name: str) -> bool:
    if not lang_name:
        return False
    return any(lang_name.startswith(pref) for pref in CASE_APPLICABLE_LANG_PREFIXES)

class TranslationChecker:
    """
    Excel Translation Checker (Gemini Single Model) - Service Version
    """

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        max_concurrency: int = 10,
        short_text_whitelist=None,
        skip_llm_when_glossary_mismatch: bool = False,
        no_backtranslation: bool = False
    ):
        # API Setup
        self.model_name = model_name
        
        # Concurrency
        if max_concurrency < 1:
            max_concurrency = 1
        self.max_concurrency = max_concurrency
        self._sem = asyncio.Semaphore(self.max_concurrency)
            
        self.no_backtranslation = no_backtranslation
        
        # Concurrency
        if max_concurrency < 1:
            max_concurrency = 1
        self.max_concurrency = max_concurrency
        self._sem = asyncio.Semaphore(self.max_concurrency)

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

        # Glossary Structure: { source_term: { 'targets': {code: term, ...}, 'rule': ... } }
        self.glossary = {}
        self.glossary_re = None # Pre-compiled regex for fast lookup
        self.glossary_map = {}  # {lower_case_term: original_case_term}
        self.glossary_headers = []
        self.source_lang_code = None

        # Integrated components
        self.model_handler = ModelHandler()
        self.bx_engine = BXGuidelineEngine()
    

    async def load_glossary_from_file(self, file_path: str, source_lang_code: str):
        """
        3행 구조의 용어집 CSV를 로드합니다.
        1행: 한국어, 영어_미국 등 (JSON 'code'와 매칭)
        2행: ko_KR, en_US 등
        3행: Lng 행
        """
        try:
            if not os.path.exists(file_path):
                return f"Glossary file not found: {file_path}"

            # 1. 헤더 3개 행을 먼저 읽어서 컬럼 인덱스 파악
            import pandas as pd
            df_headers = pd.read_csv(file_path, header=None, nrows=3, encoding='utf-8-sig').fillna("")
            
            num_cols = df_headers.shape[1]
            source_col_idx = -1
            rule_col_idx = -1
            
            # 소스 언어 컬럼 찾기 (1~3행 모두 검색)
            for c in range(num_cols):
                for r in range(3):
                    val = str(df_headers.iloc[r, c]).strip()
                    if val == source_lang_code or source_lang_code.lower() in val.lower() or val.lower() in source_lang_code.lower():
                        source_col_idx = c
                        break
                if source_col_idx != -1: break
            
            if source_col_idx == -1:
                return f"⚠ Warning: 용어집에서 '{source_lang_code}' 컬럼을 찾을 수 없습니다. (헤더 1~3행 확인 필요)"

            # 설명/규칙 컬럼 찾기
            for c in range(num_cols):
                for r in range(3):
                    val = str(df_headers.iloc[r, c]).strip()
                    if "설명" in val or "규칙" in val or "rule" in val.lower():
                        rule_col_idx = c
                        break
                if rule_col_idx != -1: break

            # 2. 데이터 로드 (4행부터)
            df_data = pd.read_csv(file_path, header=None, skiprows=3, encoding='utf-8-sig').fillna("")
            
            self.source_lang_code = source_lang_code

            count = 0
            for _, row in df_data.iterrows():
                source_term = str(row[source_col_idx]).strip()
                if not source_term or source_term.lower() == 'lng':
                    continue

                rule = str(row[rule_col_idx]).strip() if rule_col_idx != -1 else ""

                targets = {}
                for c in range(num_cols):
                    if c == source_col_idx or c == rule_col_idx:
                        continue
                    
                    # 타겟 키는 1행(영어_미국 등) 또는 2행(en_US 등)에서 가져옴
                    header_key = str(df_headers.iloc[0, c]).strip() or str(df_headers.iloc[1, c]).strip()
                    if not header_key or header_key.lower() in ['key', 'lng']:
                        continue
                        
                    val = str(row[c]).strip()
                    if val:
                        targets[header_key] = val
                        # 2행의 코드값도 키로 추가 (en_US 등)
                        code_key = str(df_headers.iloc[1, c]).strip()
                        if code_key and code_key != header_key:
                            targets[code_key] = val
                
                if targets:
                    self.glossary[source_term] = {"targets": targets, "rule": rule}
                    count += 1
            
            self._compile_glossary_re()
            return f"✓ 용어집 로드 성공: {count}개 항목 (매칭 기준: {source_lang_code})"

        except Exception as e:
            return f"Glossary load failed: {str(e)}"

    def _compile_glossary_re(self):
        """
        용어집의 소스어들을 하나의 정규표현식으로 컴파일하여 검색 성능을 최적화합니다.
        길이가 긴 단어부터 매칭되도록 정렬하여 부분 일치 오류를 방지합니다.
        """
        if not self.glossary:
            self.glossary_re = None
            return

        # 단어 길이가 긴 순서대로 정렬 (긴 단어가 우선 매칭되도록)
        sorted_terms = sorted(self.glossary.keys(), key=len, reverse=True)
        self.glossary_map = {t.lower(): t for t in sorted_terms}
        patterns = []
        
        for term in sorted_terms:
            escaped = re.escape(term)
            # 영어/숫자로 시작/종료되는 경우 단어 경계(\b)와 유사한 로직 적용 
            # (단, 한국어 조사가 붙는 경우를 고려하여 앞뒤가 영문/숫자인 경우만 경계 체크)
            pattern = escaped
            if term[0].isalnum():
                pattern = r'(?<![a-zA-Z0-9])' + pattern
            if term[-1].isalnum():
                pattern = pattern + r'(?![a-zA-Z0-9])'
            patterns.append(f"({pattern})")
            
        # 모든 패턴을 OR(|)로 결합
        try:
            self.glossary_re = re.compile('|'.join(patterns), re.IGNORECASE)
        except Exception as e:
            print(f"Regex compilation failed: {e}")
            self.glossary_re = None

    def _get_relevant_glossary_terms(self, source_text: str):
        """
        원문에서 용어집에 포함된 단어들을 추출합니다.
        """
        if not self.glossary or not self.glossary_re or not source_text:
            return []
            
        found_terms = set()
        for match in self.glossary_re.finditer(source_text):
            # 매칭된 텍스트를 소문자로 변환하여 원래 용어집 키 찾기
            matched_text = match.group(0).lower()
            original_term = self.glossary_map.get(matched_text)
            if original_term:
                found_terms.add(original_term)
        return list(found_terms)

    # ----------------- Excel Loader -----------------
    def load_excel_files(self, source_path, target_path, selected_sheets=None):
        pass

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
            
            extracted_count = 0
            try:
                # Handle full range like "A:Z" or specific "A1:C10"
                src_rows = ws_source[cell_range]
                tgt_rows = ws_target[cell_range]
                
                # Normalize to list of rows if single cell
                if not isinstance(src_rows, tuple) and not isinstance(src_rows, list): 
                    # it might be a single cell if range is "A1"
                    src_rows = ((src_rows,),)
                    tgt_rows = ((tgt_rows,),)
                elif isinstance(src_rows, tuple) and not isinstance(src_rows[0], tuple): 
                     # Single column/row tuple structure variation check
                     pass

                for source_row, target_row in zip(src_rows, tgt_rows):
                    for s_cell, t_cell in zip(source_row, target_row):
                        s_val = str(s_cell.value).strip() if s_cell.value is not None else ""
                        t_val = str(t_cell.value).strip() if t_cell.value is not None else ""
                        
                        if s_val and t_val:
                            all_data.append({
                                "cell_ref": s_cell.coordinate,
                                "sheet_name": sheet_name,
                                "source": s_val,
                                "target": t_val,
                            })
                            extracted_count += 1
            except Exception as e:
                print(f"Error accessing range {cell_range} in {sheet_name}: {e}")
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

    def _build_glossary_lines_for_code(self, target_lang_code: str, source_text: str = None):
        if not self.glossary:
            return "용어집 없음"
        
        # 원문이 주어지면 관련 용어만 필터링, 아니면 전체 (하위 호환성)
        if source_text:
            relevant_terms = self._get_relevant_glossary_terms(source_text)
        else:
            relevant_terms = list(self.glossary.keys())
            
        if not relevant_terms:
            return "관련 용어 없음"

        out = []
        for term in relevant_terms:
            meta = self.glossary[term]
            tgt = meta["targets"].get(target_lang_code)
            if not tgt:
                continue
            rule = meta.get("rule")
            rule_info = f" (규칙: {rule})" if rule else ""
            out.append(f"- 원어: {term} → 대상어({target_lang_code}): {tgt}{rule_info}")
        
        return "\n".join(out) if out else f"용어집에 '{target_lang_code}' 타겟 항목 없음"

    def _get_glossary_context_as_dict(self, target_lang_code: str, source_text: str | None = None):
        """
        용어집 데이터를 JSON/Dict 형태로 반환합니다. (프롬프트 최적화용)
        """
        if not self.glossary:
            return {}
        
        # 정규표현식이 컴파일되지 않은 경우 컴파일 시도
        if not self.glossary_re:
            self._compile_glossary_re()
            
        relevant_terms = self._get_relevant_glossary_terms(source_text) if source_text else list(self.glossary.keys())
        
        context = {}
        for term in relevant_terms:
            tgt = self.glossary[term]["targets"].get(target_lang_code)
            if tgt:
                context[term] = tgt
        return context

    def _precheck_glossary_mismatch(self, source_text: str, target_text: str, target_lang_code: str):
        if not self.glossary or not target_lang_code:
            return []
        
        relevant_terms = self._get_relevant_glossary_terms(source_text)
        if not relevant_terms:
            return []

        mismatches = []
        tgt_lower = target_text.lower()
        
        for s_term in relevant_terms:
            meta = self.glossary[s_term]
            t_term = meta["targets"].get(target_lang_code)
            if not t_term: continue
            
            if t_term.lower() not in tgt_lower:
                mismatches.append(f"'{s_term}' → '{t_term}' 미적용({target_lang_code})")
        return mismatches

    def _check_glossary_casing(self, source_text: str, target_text: str, target_lang_code: str):
        if not self.glossary or not target_lang_code or not source_text or not target_text:
            return []
            
        relevant_terms = self._get_relevant_glossary_terms(source_text)
        if not relevant_terms:
            return []

        issues = []
        tgt_lower = target_text.lower()
        
        for s_term in relevant_terms:
            meta = self.glossary[s_term]
            t_term = meta["targets"].get(target_lang_code)
            if not t_term: continue
            
            if t_term.lower() in tgt_lower:
                idx = tgt_lower.find(t_term.lower())
                if idx == -1: continue
                actual = target_text[idx:idx + len(t_term)]
                if actual.lower() == t_term.lower() and actual != t_term:
                    issues.append(f"용어집 '{t_term}'의 대소문자 표기가 '{actual}'로 사용됨")
        return issues

    def _analyze_sentence_case(self, target_text: str, target_lang: str):
        if not target_text or not _is_case_sensitive_language(target_lang):
            return None, None
        
        text = target_text.strip()
        sentences = re.split(r'(?<=[\.!?])\s+', text)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return None, None
            
        report_lines = []
        fixed_sentences = []
        
        for idx, sent in enumerate(sentences, start=1):
            s = sent
            first_alpha_index = None
            for i, ch in enumerate(s):
                if ch.isalpha():
                    first_alpha_index = i
                    break
            
            if first_alpha_index is None:
                fixed_sentences.append(s)
                continue
                
            first_char = s[first_alpha_index]
            rest = s[first_alpha_index+1:]
            
            is_sentence_case = first_char.isupper() and rest == rest.lower()
            extra_caps = sum(1 for ch in rest if ch.isalpha() and ch.isupper())
            
            status = "문장형(첫 글자만 대문자)" if is_sentence_case else "문장형 아님"
            report_lines.append(f"- 문장 {idx}: {status}, 추가 대문자 수: {extra_caps}개")
            
            fixed_rest = rest.lower()
            fixed_sent = s[:first_alpha_index] + first_char.upper() + fixed_rest
            fixed_sentences.append(fixed_sent)
            
        simple_fixed_text = " ".join(fixed_sentences)
        report = "\n".join(report_lines)
        if simple_fixed_text == target_text:
            simple_fixed_text = None
        return report, simple_fixed_text
    async def _run_llm_translation(self, text, target_lang, model_name="gemini-2.5-flash", bx_style_on=False, glossary_context=None):
        """
        내부 전용 번역 메서드: JSON 프롬프트 생성 및 LLM 호출을 담당합니다.
        """
        brackets = "[]"
        if "Japanese" in target_lang:
            brackets = "「」"
        elif "Chinese" in target_lang:
            brackets = "【】"

        # 입력 데이터 구조화
        input_data = {
            "source_text": text,
            "target_language": target_lang,
            "glossary": glossary_context if isinstance(glossary_context, dict) else {},
            "formatting": {
                "glossary_prefix": brackets[0],
                "glossary_suffix": brackets[1]
            }
        }

        if bx_style_on and self.bx_engine:
            system_prompt = self.bx_engine.get_system_prompt(target_lang)
            system_prompt += (
                f"\nIMPORTANT: You must follow the Samsung BX style guide above.\n"
                f"Use the provided glossary if available. CRITICAL: Any time you use a term from the glossary, you MUST wrap it in '{brackets[0]}' and '{brackets[1]}' (e.g., {brackets[0]}Term{brackets[1]}).\n"
                "Return the response in JSON format with a 'translation' field."
            )
        else:
            system_prompt = (
                f"You are a professional {target_lang} localizer.\n"
                f"TASK: Translate the source text naturally for a native speaker while preserving 100% of the original meaning.\n"
                f"RULE 1: Use the provided 'glossary'.\n"
                f"RULE 2: Wrap glossary terms in '{brackets[0]}' and '{brackets[1]}'. (CRITICAL)\n"
                "OUTPUT: Return ONLY a JSON object with a 'translation' key."
            )

        try:
            prompt = json.dumps(input_data, ensure_ascii=False, indent=2)
            response_data = await self.model_handler.generate_content(
                prompt, 
                model_name=model_name, 
                system_instruction=system_prompt,
                response_json=True
            )

            if isinstance(response_data, dict):
                return response_data.get("translation", str(response_data))
            return str(response_data)
        except Exception as e:
            return f"[번역 오류] {str(e)}"

    async def _run_llm_audit(self, source_text, translated_text, target_lang, model_name="gpt-5-mini"):
        """
        내부 전용 감사 메서드: JSON 구조화된 프롬프트를 사용하여 품질 검수합니다.
        """
        input_data = {
            "source_text": source_text,
            "translated_text": translated_text,
            "language": target_lang
        }

        system_instruction = "당신은 원문과 번역문을 대조하여 언어적 정확성과 품질을 평가하는 전문 검수자입니다. 분석 결과를 한국어 의견으로 반환하세요."
        
        try:
            prompt = json.dumps(input_data, ensure_ascii=False, indent=2)
            response = await self.model_handler.generate_content(
                prompt,
                model_name=model_name,
                system_instruction=system_instruction + "\nReturn a JSON object with 'reasoning' and 'is_accurate' (bool) keys.",
                response_json=True
            )
            
            if isinstance(response, dict):
                return response.get("reasoning", str(response))
            return str(response)
        except Exception as e:
            return f"[감사 오류] {str(e)}"


    # ----------------- LLM Calls -----------------
    async def check_with_llm_qa(self, source_text, target_text, source_lang, target_lang, target_lang_code):
        glossary_dict = self._get_glossary_context_as_dict(target_lang_code, source_text=source_text)
        
        # QA용 구조화된 프롬프트 데이터
        input_data = {
            "source": {"lang": source_lang, "text": source_text},
            "translation": {"lang": f"{target_lang}/{target_lang_code}", "text": target_text},
            "glossary": glossary_dict
        }

        prompt = f"""당신은 전문 번역 검수 전문가입니다. 아래 JSON 데이터를 분석하여 상세한 검수 결과를 반환하세요.

[Input Data]
{json.dumps(input_data, ensure_ascii=False, indent=2)}

[검수 가이드라인]
1. 문법/유창성: 오타, 문법 오류, 성수 일치, 관용구 사용 등 정밀 점검. 
2. 정확성: 원문의 뉘앙스(예: 공손함의 정도)와 정보가 정확히 전달되었는지 확인.
3. 용어집 준수: 제공된 glossary 데이터와 100% 일치하는지 확인 (대소문자, 띄어쓰기 포함).
4. 대소문자(Casing): 해당 언어의 문장형(Sentence case) 또는 타이틀형(Title case) 등 일반 규칙 준수 여부.
5. 품질 등급 평가: 번역의 품질을 다각도로 분석하여 기술하세요.

[출력 형식]
JSON 형식으로 반환하세요:
{{
  "evaluation": "항목별 상세 점검 결과 (한국어 불렛 포인트로 풍부하게 작성)",
  "is_excellent": true/false (중대한 결함이 없을 경우 true),
  "suggested_fix": "가장 자연스럽고 정확한 수정 제안 (필요 시)"
}}
"""
        try:
            # ModelHandler가 JSON 모드를 지원하므로 딕셔너리로 바로 받을 수 있음
            response = await self.model_handler.generate_content(
                prompt, 
                model_name=self.model_name, 
                system_instruction="당신은 언어별 문법과 문맥을 정밀하게 분석하는 전문 번역 검수자입니다. 반드시 JSON 형식으로만 응답합니다.",
                response_json=True
            )
            
            if isinstance(response, str):
                 return f"[QA 결과]\n{response}"
            
            eval_text = response.get("evaluation", "검수 결과 파싱 실패")
            if isinstance(eval_text, list):
                # Ensure all items are strings (e.g., handle if AI returns a list of dicts)
                eval_text = "\n".join(
                    json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
                    for item in eval_text
                )
            elif isinstance(eval_text, dict):
                eval_text = json.dumps(eval_text, ensure_ascii=False, indent=2)
            
            if response.get("is_excellent"):
                eval_text += "\n\n최종 평가: 우수, 주요 문제 없음."
            
            if response.get("suggested_fix"):
                eval_text += f"\n\n[수정안 제안]:\n{response['suggested_fix']}"
                
            return eval_text
        except Exception as e:
            return f"[QA 오류]: {e}"






    async def get_back_translation(self, target_text, target_lang, source_lang):
        prompt = (
            f"다음 {target_lang} 텍스트를 {source_lang}으로 다시 번역해주세요. "
            f"오직 번역된 텍스트만 제공해야 합니다. 다른 설명이나 텍스트는 포함하지 마세요.\n\n{target_text}"
        )
        try:
            return await self.model_handler.generate_content(prompt, model_name=self.model_name)
        except Exception as e:
            return f"[역번역 오류]: {e}"

    # ----------------- Process Single Item -----------------
    async def process_item(self, item, source_lang, default_target_lang, sheet_lang_map, default_target_lang_code):
        cell_ref = item["cell_ref"]
        sheet_name = item["sheet_name"]
        source = item["source"]
        target = item["target"]
        
        tgt_lang = sheet_lang_map.get(sheet_name, {}).get("lang", default_target_lang)
        tgt_code = sheet_lang_map.get(sheet_name, {}).get("code", default_target_lang_code)
        
        # Skip detection
        is_placeholder = (
            (len(source) <= 2 and len(target) <= 2 and source.lower() == target.lower()) or
            (not source.strip() and not target.strip()) or
            (source.strip().lower() == target.strip().lower() and len(source.strip()) < 10)
        )
        if source.strip().lower() in self.short_text_whitelist or target.strip().lower() in self.short_text_whitelist:
            is_placeholder = False
        
        pre_mismatch = self._precheck_glossary_mismatch(source, target, tgt_code)
        skip_llm = self.skip_llm_when_glossary_mismatch and bool(pre_mismatch)
        
        case_report, simple_case_fix = self._analyze_sentence_case(target, tgt_lang)
        glossary_case_issues = self._check_glossary_casing(source, target, tgt_code)
        
        # Construct Partial Report Sections
        case_section = "대소문자 하드룰(문장형) 점검:\n" + case_report if case_report else "별도 지적 사항 없음."
        if simple_case_fix:
            case_section += f"\n\n[단순 규칙 기반 문장형 변환안]:\n{simple_case_fix}"

        glossary_parts = []
        if pre_mismatch: glossary_parts.append("용어집 사전 감지:\n- " + "\n- ".join(pre_mismatch))
        if glossary_case_issues: glossary_parts.append("용어집 대소문자 표기 점검:\n" + "\n".join(f"- {msg}" for msg in glossary_case_issues))
        glossary_section = "\n\n".join(glossary_parts) if glossary_parts else "별도 지적 사항 없음."

        # Result container
        res = {
            "sheet_name": sheet_name,
            "cell_ref": cell_ref,
            "source": source,
            "target": target,
            "case_section": case_section,
            "glossary_section": glossary_section,
            "back_translation": "",
            "ai_review": ""
        }

        # Skip Logic
        if is_placeholder and not pre_mismatch:
            res["back_translation"] = "[건너뜀: 짧은/무의미]"
            res["ai_review"] = "[건너뜀: 짧은/무의미]"
            return res

        # LLM Logic
        if skip_llm:
            res["back_translation"] = "[사전 감지로 LLM 호출 생략]"
            res["ai_review"] = "※ 용어집 사전 감지 결과를 우선 검토하세요."
        else:
            # LLM Calls
            qa_task = self._with_semaphore(
                self.check_with_llm_qa(source, target, source_lang, tgt_lang, tgt_code)
            )
            
            if self.no_backtranslation:
                res["ai_review"] = await qa_task
                res["back_translation"] = "[역번역 비활성화됨]"
            else:
                bt_task = self._with_semaphore(
                    self.get_back_translation(target, tgt_lang, source_lang)
                )
                review, bt = await asyncio.gather(qa_task, bt_task)
                res["ai_review"] = review
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
        source_sheet_name: str = None
    ):
        """
        Yields events:
        {"type": "log", "message": "..."}
        {"type": "progress", "current": n, "total": m, "percent": ...}
        {"type": "result_chunk", "data": formatted_string}
        {"type": "complete", "total": m, "output_data": all_text}
        """
        yield {"type": "log", "message": "용어집 로드 시작..."}
        
        # If source_sheet_name is provided, use it to infer source_lang
        if source_sheet_name and sheet_lang_map:
            source_info = sheet_lang_map.get(source_sheet_name)
            if source_info:
                source_lang = source_info.get('lang', source_lang)
        
        # Load Glossary
        if glossary_file_path:
             yield {"type": "log", "message": f"용어집 파일 로드 중: {os.path.basename(glossary_file_path)}"}
             msg = await self.load_glossary_from_file(glossary_file_path, source_lang)
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
            yield {"type": "complete", "total": 0, "output_data": "검수할 데이터가 없습니다."}
            return

        # Wrapper to track index
        async def process_with_index(index, item):
            res = await self.process_item(item, source_lang, target_lang, sheet_lang_map, target_lang_code)
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
                fmt_result = (
                    f"\n\n{'='*90}\n"
                    f"[시트] {res['sheet_name']} | [셀] {res['cell_ref']}\n"
                    f"{'-'*90}\n\n"
                    f"[상세 - 원문]\n{res['source']}\n\n"
                    f"[상세 - 번역문]\n{res['target']}\n\n"
                    f"[상세 - 대소문자 점검]\n{res['case_section']}\n\n"
                    f"[상세 - 용어집 점검]\n{res['glossary_section']}\n\n"
                    f"[상세 - 역번역]\n{res['back_translation']}\n\n"
                    f"[상세 - ai 검수 결과]\n{res['ai_review']}\n"
                    f"{'='*90}\n"
                )
                
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

        # Final Header/Footer assembly using ORDERED results
        final_text = []
        header = (
            f"--- 번역 검수 보고서 (Model: {self.model_name}) ---\n"
            f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"총 검수 항목: {total_items}개\n"
            f"용어집 항목: {len(self.glossary)}\n\n"
        )
        final_text.append(header)
        # Filter out None results if any failed
        valid_results = [r for r in ordered_results if r is not None]
        final_text.extend(valid_results)
        
        full_output = "".join(final_text)
        yield {"type": "complete", "total": total_items, "output_data": full_output}

    # ----------------- Integrated Translation & Audit Generator -----------------
    async def run_integrated_pipeline_generator(
        self,
        source_file_path,
        cell_range,
        bx_style_on,
        sheet_lang_map,
        translation_model="gemini-2.5-flash",
        audit_model="gpt-5.2",
        glossary_file_path=None,
        selected_sheets=None,
        source_sheet_name=None,
        skip_audit=False
    ):
        """
        Translates source text, audits it, and saves it to a NEW Excel file.
        Yields SSE events same as inspection.
        """
        yield {"type": "log", "message": "Starting Integrated Translation & Audit Pipeline..."}
        
        # 1. Load Excel
        try:
            wb = openpyxl.load_workbook(source_file_path)
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
            yield {"type": "log", "message": f"Loading glossary (Source: {source_lang})..."}
            msg = await self.load_glossary_from_file(glossary_file_path, source_lang)
            yield {"type": "log", "message": msg}

        source_ws = wb[source_sheet_name]
        
        # 3. Extract source data
        source_data = []
        rows = source_ws[cell_range]
        if not isinstance(rows, tuple) and not isinstance(rows, list):
            rows = ((rows,),)
        for row in rows:
            for cell in row:
                if cell.value:
                    source_data.append({'text': str(cell.value).strip(), 'coord': cell.coordinate})

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
            
            # Step 1: Translate
            glossary_dict = self._get_glossary_context_as_dict(target_lang_code, source_text=source_text)
            if not glossary_dict:
                glossary_dict = None

            translation = await self._run_llm_translation(
                source_text, 
                target_lang, 
                model_name=translation_model,
                bx_style_on=bx_style_on,
                glossary_context=glossary_dict
            )
            
            ws[coord].value = translation

            item_data = {"cell_ref": coord,"sheet_name": ws.title,"source": source_text,"target": translation}
            
            if skip_audit:
                res = {
                    "sheet_name": ws.title,
                    "cell_ref": coord,
                    "source": source_text,
                    "target": translation,
                    "case_section": "[Bypassed]",
                    "glossary_section": "[Bypassed]",
                    "back_translation": "[Bypassed]",
                    "ai_review": "[Bypassed: Translate Only Mode]"
                }
            else:
                res = await self.process_item(item_data, source_lang, target_lang, sheet_lang_map, target_lang_code)
            
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
        
        # Yield per-cell progress
        for future in asyncio.as_completed(tasks):
            try:
                index, res = await future
                completed_count += 1
                
                fmt_result = (
                    f"\n\n{'='*90}\n"
                    f"[시트] {res['sheet_name']} | [셀] {res['cell_ref']}\n"
                    f"{'-'*90}\n\n"
                    f"[상세 - 원문]\n{res['source']}\n\n"
                    f"[상세 - 번역문]\n{res['target']}\n\n"
                    f"[상세 - 대소문자 점검]\n{res['case_section']}\n\n"
                    f"[상세 - 용어집 점검]\n{res['glossary_section']}\n\n"
                    f"[상세 - 역번역]\n{res['back_translation']}\n\n"
                    f"[상세 - ai 검수 결과]\n{res['ai_review']}\n"
                    f"{'='*90}\n"
                )
                ordered_results[index] = fmt_result
                
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
        wb.save(out_excel_path)
        
        header = (
            f"--- 번역 통합 검수 보고서 (Model: {self.model_name}) ---\n"
            f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"총 항목: {completed_count}개\n\n"
        )
        valid_results = [r for r in ordered_results if r is not None]
        report_text = header + "".join(valid_results)
        
        yield {
            "type": "complete", 
            "output_data": report_text, 
            "excel_path": out_excel_path
        }

