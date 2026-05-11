# -*- coding: utf-8 -*-
"""
Compatibility wrapper for Samsung BX prompts.

The prompt definitions now live in prompt_modules.py and are assembled by
PromptBuilder. This wrapper keeps older imports working.
"""

from prompt_builder import PromptBuilder


class BXGuidelineEngine:
    def __init__(self):
        self.prompt_builder = PromptBuilder()

    def get_system_prompt(self, target_lang):
        return self.prompt_builder.build_translation_prompt(
            target_lang=target_lang,
            bx_style_on=True,
        )

    def get_audit_prompt(self, source_text, translated_text, target_lang):
        return self.prompt_builder.build_bx_audit_prompt(
            source_text=source_text,
            translated_text=translated_text,
            target_lang=target_lang,
        )
