import os
import json
import re
from datetime import datetime
from google.genai import types
from openai import AsyncOpenAI
from dotenv import load_dotenv

from translation_web_app.gemini_auth import build_gemini_client

load_dotenv()

class ModelHandler:
    def __init__(self, gemini_api_key: str = None, openai_api_key: str = None):
        # OpenAI config — runtime key takes priority over .env
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if self.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)

        # Gemini config (New SDK) — runtime key takes priority over .env
        # Vertex AI(서비스 계정)가 우선이고, 평문 API Key는 fallback (build_gemini_client 참고)
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_client = None
        self.is_vertex_ai = False
        try:
            self.gemini_client, self.is_vertex_ai = build_gemini_client(self.gemini_api_key)
        except ValueError:
            pass

        # 모델별 호출/토큰 사용량 누적 (Thinking/Reasoning 토큰 포함)
        self.usage_stats: dict[str, dict] = {}

    def _record_usage(self, model_name: str, provider: str, input_tokens: int = 0,
                       output_tokens: int = 0, thinking_tokens: int = 0, cached_tokens: int = 0):
        entry = self.usage_stats.setdefault(model_name, {
            "provider": provider, "calls": 0, "input_tokens": 0,
            "output_tokens": 0, "thinking_tokens": 0, "cached_tokens": 0,
        })
        entry["calls"] += 1
        entry["input_tokens"] += input_tokens
        entry["output_tokens"] += output_tokens
        entry["thinking_tokens"] += thinking_tokens
        entry["cached_tokens"] += cached_tokens

    def get_usage_report(self) -> str:
        """검수 리포트 최상단에 붙일 평문 사용량 요약 (호출 없으면 빈 문자열)."""
        gemini_models = {k: v for k, v in self.usage_stats.items() if v["provider"] == "gemini"}
        gpt_models = {k: v for k, v in self.usage_stats.items() if v["provider"] == "openai"}
        if not gemini_models and not gpt_models:
            return ""

        lines = [
            "--- AI 사용량 및 모델 정보 ---",
            f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        def _section(title, models, thinking_label):
            for name, s in models.items():
                lines.append(f"[{title}] {name}")
                parts = [
                    f"호출 {s['calls']}회",
                    f"입력 {s['input_tokens']:,} tok",
                    f"출력 {s['output_tokens']:,} tok",
                    f"{thinking_label} {s['thinking_tokens']:,} tok",
                ]
                if s["cached_tokens"]:
                    parts.append(f"캐시 {s['cached_tokens']:,} tok")
                lines.append("  " + " | ".join(parts))
                lines.append("")

        if gemini_models:
            _section("Gemini", gemini_models, "Thinking")
        if gpt_models:
            _section("OpenAI", gpt_models, "Reasoning")

        return "\n".join(lines) + "\n"

    def _safe_parse_json(self, text: str):
        """
        AI가 반환한 텍스트에서 JSON 블록을 찾아 파싱합니다.
        마크다운 백틱(```json ... ```)이나 불필요한 서술어를 처리합니다.
        """
        if not text:
            return {}
        try:
            # 1. 마크다운 스타일 JSON 블록 추출
            clean_text = text.strip()
            if "```json" in clean_text:
                clean_text = clean_res = clean_text.split("```json")[-1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[-1].split("```")[0].strip()
            
            # 2. { } 외부의 텍스트 제거 시도 (가장 바깥쪽 중괄호 찾기)
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                clean_text = str(clean_text)[start_idx:end_idx + 1]
            
            return json.loads(clean_text)
        except Exception as e:
            print(f"JSON Parsing Error: {e}\nOriginal Text: {text}")
            return {"error": "parsing_failed", "original_text": text}

    async def call_gemini(self, prompt, model_name="gemini-2.5-flash", system_instruction=None,
                           response_json=False, thinking_budget: int | None = None):
        if not self.gemini_client:
            return "Gemini API Key not configured."

        try:
            actual_model_name = model_name
            if not self.is_vertex_ai:
                if not actual_model_name.startswith("models/"):
                    actual_model_name = f"models/{actual_model_name}"

            config_kwargs = {}
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction
            if response_json:
                config_kwargs["response_mime_type"] = "application/json"
            if thinking_budget is not None:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
            config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

            response = await self.gemini_client.aio.models.generate_content(
                model=actual_model_name,
                contents=prompt,
                config=config
            )

            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                self._record_usage(
                    model_name,
                    provider="gemini",
                    input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                    output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                    thinking_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
                    cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
                )

            res_text = response.text.strip()
            if response_json:
                return self._safe_parse_json(res_text)
            return res_text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    async def call_gpt(self, prompt, model_name="gpt-5.4-mini", system_instruction=None,
                        response_json=False, reasoning_effort: str | None = None):
        if not self.openai_client:
            return "OpenAI API Key not configured."

        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})

            # OpenAI JSON 모드는 프롬프트에 'json' 단어가 포함되어야 함
            final_prompt = prompt
            if response_json and "json" not in prompt.lower():
                final_prompt += "\n\nReturn the output in JSON format."

            messages.append({"role": "user", "content": final_prompt})

            kwargs = {
                "model": model_name,
                "messages": messages
            }
            if response_json:
                kwargs["response_format"] = {"type": "json_object"}
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

            response = await self.openai_client.chat.completions.create(**kwargs)
            res_text = response.choices[0].message.content.strip()

            usage = getattr(response, "usage", None)
            if usage is not None:
                details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details is not None else 0
                self._record_usage(
                    model_name,
                    provider="openai",
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    thinking_tokens=reasoning_tokens or 0,
                )

            if response_json:
                return self._safe_parse_json(res_text)
            return res_text
        except Exception as e:
            return f"GPT Error: {str(e)}"

    async def generate_content(self, prompt, model_name="gemini-2.5-flash", system_instruction=None,
                                response_json=False, thinking_budget: int | None = None,
                                reasoning_effort: str | None = None):
        """Unified method to call either Gemini or GPT based on model name."""
        if any(k in model_name.lower() for k in ("gpt", "o1", "o3", "o4")):
            return await self.call_gpt(prompt, model_name=model_name, system_instruction=system_instruction,
                                        response_json=response_json, reasoning_effort=reasoning_effort)
        else:
            return await self.call_gemini(prompt, model_name=model_name, system_instruction=system_instruction,
                                           response_json=response_json, thinking_budget=thinking_budget)

    async def count_tokens(self, text: str, model_name: str = "gemini-2.5-flash") -> tuple[int, str]:
        """
        Returns (token_count, method).
        method: 'gemini_api' | 'tiktoken' | 'estimated'
        GPT/o-series  → tiktoken (local, free).
        Gemini        → client.models.count_tokens (lightweight, ~free).
        Fallback      → CJK/1.5 + rest/4 estimation.
        """
        import asyncio as _asyncio

        is_gpt = any(k in model_name.lower() for k in ("gpt", "o1", "o3", "o4"))

        if is_gpt:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("o200k_base")
                return len(enc.encode(text)), "tiktoken"
            except Exception:
                pass
        else:
            if self.gemini_client:
                try:
                    actual = model_name if model_name.startswith("models/") or self.is_vertex_ai else f"models/{model_name}"
                    result = await _asyncio.to_thread(
                        self.gemini_client.models.count_tokens,
                        model=actual,
                        contents=text,
                    )
                    return result.total_tokens, "gemini_api"
                except Exception:
                    pass
            try:
                import tiktoken
                enc = tiktoken.get_encoding("o200k_base")
                return len(enc.encode(text)), "tiktoken"
            except Exception:
                pass

        cjk = len(re.findall(r'[　-鿿豈-﫿가-힯]', text))
        rest = len(text) - cjk
        return round(cjk / 1.5 + rest / 4), "estimated"
