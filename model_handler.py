import os
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class ModelHandler:
    def __init__(self):
        # Gemini config (New SDK)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = None
        if self.gemini_api_key:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        
        # OpenAI config
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if self.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)

    async def call_gemini(self, prompt, model_name="gemini-2.5-flash", system_instruction=None):
        if not self.gemini_client:
            return "Gemini API Key not configured."
        
        try:
            # Ensure model name doesn't have double 'models/' prefix
            actual_model_name = model_name
            if not actual_model_name.startswith("models/"):
                actual_model_name = f"models/{actual_model_name}"
            
            config = None
            if system_instruction:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            
            response = await self.gemini_client.aio.models.generate_content(
                model=actual_model_name,
                contents=prompt,
                config=config
            )
            return response.text.strip()
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    async def call_gpt(self, prompt, model_name="gpt-5-mini", system_instruction=None):
        if not self.openai_client:
            return "OpenAI API Key not configured."
        
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"GPT Error: {str(e)}"

    async def generate_content(self, prompt, model_name="gemini-2.5-flash", system_instruction=None):
        """Unified method to call either Gemini or GPT based on model name."""
        if "gpt" in model_name.lower() or "o1" in model_name.lower():
            return await self.call_gpt(prompt, model_name=model_name, system_instruction=system_instruction)
        else:
            return await self.call_gemini(prompt, model_name=model_name, system_instruction=system_instruction)

    async def translate(self, text, target_lang, model_name="gemini-2.5-flash", bx_handler=None, bx_style_on=False, glossary_context=None):
        glossary_prompt = ""
        if glossary_context:
            glossary_prompt = f"\n\n[Glossary Rules]\n{glossary_context}\nYou MUST follow the glossary rules above."

        # Determine brackets for glossary terms
        brackets = "[]"
        if "Japanese" in target_lang:
            brackets = "「」"
        elif "Chinese" in target_lang:
            brackets = "【】"

        if bx_style_on and bx_handler:
            system_prompt = bx_handler.get_system_prompt(target_lang)
            # Add strict instruction to system prompt for BX
            system_prompt += (
                "\nIMPORTANT: Return ONLY the translated text. No explanations, no alternatives, no titles.\n"
                f"If glossary terms are provided, you MUST use them and wrap them in '{brackets[0]}' and '{brackets[1]}'."
            )
            return await self.generate_content(text + glossary_prompt, model_name=model_name, system_instruction=system_prompt)
        else:
            prompt = (
                f"Translate the following text to {target_lang}.{glossary_prompt}\n"
                f"If glossary rules are provided, you MUST use the provided translations and wrap those specific terms in '{brackets[0]}' and '{brackets[1]}'.\n"
                "Return ONLY the translation. Ensure no preamble, no markdown, no quotes, no alternative suggestions. 1:1 correspondence only.\n\n"
                f"Text: {text}"
            )
            return await self.generate_content(prompt, model_name=model_name)

    async def audit(self, source_text, translated_text, target_lang, model_name="gpt-5-mini", bx_handler=None):
        # NOTE: IntegratedTranslationService now uses TranslationChecker directly for auditing.
        # This method is kept for legacy compatibility but updated to use gpt-5-mini.
        if bx_handler:
            prompt = bx_handler.get_audit_prompt(source_text, translated_text, target_lang)
            return await self.call_gpt(prompt, model_name=model_name)
        else:
            prompt = f"""Audit the following translation for accuracy and nuance. 
Source: {source_text}
Translation: {translated_text}
Language: {target_lang}
Provide reasoning in Korean."""
            return await self.call_gpt(prompt, model_name=model_name)
