# -*- coding: utf-8 -*-
"""
Prompt builder for translation and audit prompts.

The public methods preserve the existing JSON response contracts while making
prompt modules visible and reusable.
"""

import re

from prompt_modules import (
    AUDIT_CHECKLIST_RULES,
    AUDIT_GRADE_CRITERIA,
    AUDIT_INTRO,
    BX_STYLE_RULES,
    COMMON_LOCALIZATION_STANDARD,
    GLOSSARY_BRACKET_WRAP_RULE,
    GLOSSARY_DISCLAIMER_NAV_EXCEPTION,
    GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE,
    GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_INTL,
    GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_JA,
    GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_US,
    GLOSSARY_EXEMPT_MARKERS,
    GLOSSARY_NO_BRACKET_INSTRUCTION,
    GLOSSARY_TERM_RULES,
    LANGUAGE_LOCALIZATION_RULES,
    TYPOGRAPHY_AND_PUNCTUATION_RULES,
)


_LANGUAGE_RULE_LABELS = {
    "Korean": "Korean Honorifics & Style Consistency",
    "English": "US English Consistency",
    "English_US": "US English Consistency",
    "English_UK": "British English Consistency",
    "English_AU": "Australian English Consistency",
    "English_SG": "Singapore English Consistency",
    "German": "German Du-form Consistency",
    "Japanese": "Japanese ます-form Consistency",
    "French": "French Tone and Consistency",
    "French_BE": "Belgian French Consistency",
    "French_CA": "Canadian French Consistency",
    "Italian": "Italian UI Phrasing Consistency",
    "Spanish": "Spanish Usted and Regional Consistency",
    "Spanish_ES": "Spain Spanish Consistency",
    "Dutch": "Dutch Directness & Phrasing",
    "Swedish": "Swedish UI Phrasing & Case Consistency",
    "Arabic": "MSA & Arabic UI Conventions",
    "Brazilian Portuguese": "Brazilian Portuguese Consistency",
    "European Portuguese": "European Portuguese Consistency",
    "Russian": "Russian Word Order & Phrasing",
    "Turkish": "Turkish UI Phrasing Consistency",
    "Simplified Chinese": "Simplified Chinese Consistency",
    "Traditional Chinese": "Traditional Chinese Consistency",
    "Polish": "Polish Grammar & Phrasing Consistency",
    "Vietnamese": "Vietnamese Phrasing & Case Consistency",
    "Thai": "Thai UI Phrasing & Punctuation",
    "Indonesian": "Indonesian Phrasing & Style Consistency",
}


class PromptBuilder:
    def get_language_rule(self, target_lang: str):
        if not target_lang:
            return None
        normalized = target_lang.lower()
        for lang_key, rule in LANGUAGE_LOCALIZATION_RULES.items():
            if lang_key.lower() == normalized:
                return lang_key, rule
        for lang_key, rule in sorted(LANGUAGE_LOCALIZATION_RULES.items(), key=lambda item: len(item[0]), reverse=True):
            if lang_key.lower() in normalized:
                return lang_key, rule
        return None

    def get_brackets(self, target_lang: str) -> str:
        if target_lang and ("Japanese" in target_lang or "일본" in target_lang):
            return "「」"
        return "[]"

    def get_exempt_markers(self) -> list[str]:
        return GLOSSARY_EXEMPT_MARKERS

    def get_glossary_context_mode(self, row_key: str) -> str:
        key = (row_key or "").strip().lower()
        if "disclaimer" in key:
            return "disclaimer"
        if "description" in key:
            return "description"
        if "title" in key or "button" in key or bool(re.search(r"\d+$", key)):
            return "title_button"
        return "description"

    def _is_us_english(self, target_lang: str, target_lang_code: str = "") -> bool:
        combined = f"{target_lang} {target_lang_code}".lower().replace("-", "_")
        is_english = "english" in combined or "영어" in combined or "en_" in combined
        is_us = "us" in combined or "미국" in combined or "american" in combined
        return is_english and is_us

    def should_skip_brackets(self, row_key: str) -> bool:
        return self.get_glossary_context_mode(row_key) == "title_button"

    def should_wrap_glossary(self, row_key: str, rule_text: str = "") -> bool:
        """Single source of truth for whether a term should be wrapped in brackets."""
        if self.should_skip_brackets(row_key):
            return False
        
        if rule_text:
            clean_rule = rule_text.lower()
            if any(marker in clean_rule for marker in self.get_exempt_markers()):
                return False
                
        return True

    def build_input_formatting(self, target_lang: str, row_key: str = "") -> dict:
        if self.get_glossary_context_mode(row_key) == "title_button":
            return {"glossary_prefix": "", "glossary_suffix": ""}
        brackets = self.get_brackets(target_lang)
        return {
            "glossary_prefix": brackets[0],
            "glossary_suffix": brackets[1],
        }

    def build_translation_prompt(
        self,
        target_lang: str,
        source_lang: str = "English",
        bx_style_on: bool = False,
        rag_context: str | None = None,
        row_key: str = "",
        glossary_context=None,
        target_lang_code: str = "",
    ) -> str:
        sections = []
        sections.append(self._build_persona_section(target_lang, source_lang, bx_style_on))
        sections.append(self._build_common_section())

        language_section = self._build_language_section(target_lang)
        if language_section:
            sections.append(language_section)

        if bx_style_on:
            sections.append(self._build_bx_section(target_lang))

        if rag_context:
            rag_section = self._normalize_rag_section(rag_context)
            sections.append(
                f"{rag_section}\n"
                "Use these examples as style and terminology reference to maintain consistency."
            )

        sections.append(self._build_formatting_section(target_lang, row_key, target_lang_code, bool(glossary_context)))
        sections.append('OUTPUT: Return ONLY a JSON object with a "translation" key.')

        return "\n\n".join(section for section in sections if section).strip()

    def build_audit_prompt(
        self,
        source_lang: str,
        target_lang: str,
        target_lang_code: str | None = None,
        row_key: str = "",
        glossary_context=None,
    ) -> str:
        sections = [AUDIT_INTRO]

        language_section = self._build_language_section(target_lang, korean_heading=True)
        if language_section:
            sections.append(language_section)

        sections.append(self._build_formatting_section(target_lang, row_key, target_lang_code or "", bool(glossary_context)))
        sections.append(self._build_audit_checklist())

        if glossary_context:
            sections.append(f"[Glossary Target]\nTarget code: {target_lang_code or target_lang}")

        sections.append(self._build_audit_output_format())
        return "\n\n".join(sections).strip()

    def build_bx_audit_prompt(self, source_text: str, translated_text: str, target_lang: str) -> str:
        identity = BX_STYLE_RULES["system_identity"]
        return f"""You are a Samsung BX Audit Expert.
Evaluate if the following translation aligns with the Samsung BX Persona and Voice Attributes.

Source: {source_text}
Translation: {translated_text}
Target Language: {target_lang}

Persona: {identity['persona']}
Voice Attributes to check:
- OPEN: Use of wit, metaphor, or personification. Short, rhythmic "Double Take" headlines.
- BOLD: Confidence, contrast, and impact. No hedging words.
- AUTHENTIC: Relatable, friendly, and positive reframing.

Provide your reasoning in Korean. Specifically explain WHY this expression is suitable for Samsung's brand tone, or suggest improvements if it fails.
If it adheres well, start with [PASS]. If it needs improvement, start with [FAIL].
"""

    def describe_applied_modules(
        self,
        target_lang: str,
        source_lang: str = "English",
        bx_style_on: bool = False,
        rag_available: bool = False,
        glossary_available: bool = False,
        row_key: str = "",
    ) -> dict:
        language_match = self.get_language_rule(target_lang)
        lang_key = language_match[0] if language_match else None
        language_rules = language_match[1] if language_match else None
        glossary_context_mode = self.get_glossary_context_mode(row_key)
        bracket_mode = "no_brackets" if glossary_context_mode == "title_button" else f"wrap ({glossary_context_mode})"
        brackets = self.get_brackets(target_lang)

        return {
            "common": {
                "active": True,
                "name": COMMON_LOCALIZATION_STANDARD["name"],
                "description": "Meaning preservation, natural local expression, cultural fit, tone, and risky wording controls.",
            },
            "language": {
                "active": bool(language_rules),
                "name": _LANGUAGE_RULE_LABELS.get(lang_key, lang_key) if lang_key else "No language-specific module",
                "description": "; ".join(language_rules) if language_rules else "Only the common localization standard is applied.",
            },
            "bx": {
                "active": bool(bx_style_on),
                "name": "Samsung BX Style",
                "description": "Confident Explorer persona and OPEN/BOLD/AUTHENTIC voice rules." if bx_style_on else "BX style is off for this run.",
            },
            "formatting": {
                "active": True,
                "name": "Format and Glossary Rules",
                "description": f"Context mode: {glossary_context_mode}; bracket mode: {bracket_mode}; brackets: {brackets}; navigation path rules applied.",
            },
            "typography": {
                "active": True,
                "name": TYPOGRAPHY_AND_PUNCTUATION_RULES["name"],
                "description": "Target-locale punctuation, spacing, quotation marks, and sentence-ending style are enforced.",
            },
            "glossary": {
                "active": bool(glossary_available),
                "name": "Glossary Matching",
                "description": "Uploaded glossary terms and rule/remark exceptions are applied." if glossary_available else "No glossary file is currently selected.",
            },
            "rag": {
                "active": bool(rag_available),
                "name": "RAG Translation Memory",
                "description": "Similar previous translation examples are injected when available." if rag_available else "RAG is unavailable or empty.",
            },
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

    def _build_persona_section(self, target_lang: str, source_lang: str, bx_style_on: bool) -> str:
        if bx_style_on:
            role = BX_STYLE_RULES["system_identity"]["role"]
            return (
                f"You are the {role}.\n"
                f"Source Language: {source_lang}\n"
                f"Target Language: {target_lang}\n"
                "TASK: Translate and polish the source text naturally for a native speaker while preserving 100% of the original meaning."
            )
        return (
            f"You are a professional {target_lang} localizer.\n"
            f"Source Language: {source_lang}\n"
            f"Target Language: {target_lang}\n"
            "TASK: Translate the source text naturally for a native speaker while preserving 100% of the original meaning."
        )

    def _build_common_section(self, korean_heading: bool = False) -> str:
        heading = "[공통 현지화 품질 기준]" if korean_heading else "[COMMON LOCALIZATION STANDARD]"
        return heading + "\n" + "\n".join(f"- {rule}" for rule in COMMON_LOCALIZATION_STANDARD["rules"])

    def _build_language_section(self, target_lang: str, korean_heading: bool = False) -> str:
        match = self.get_language_rule(target_lang)
        if not match:
            return ""
        lang_key, rules = match
        heading = "[언어별 현지화 기준]" if korean_heading else "[LANGUAGE SPECIFIC RULE]"
        label = _LANGUAGE_RULE_LABELS.get(lang_key, lang_key)
        return f"{heading}\n{label}\n" + "\n".join(f"- {item}" for item in rules)

    def _build_bx_section(self, target_lang: str) -> str:
        identity = BX_STYLE_RULES["system_identity"]
        voice = BX_STYLE_RULES["voice_attributes"]
        lines = [
            "[SAMSUNG BX STYLE]",
            f"Persona: {identity['persona']}",
            f"Traits: {', '.join(identity['traits'])}",
            f"Goal: {identity['goal']}",
            f"Target Language: {target_lang}",
            "",
            "Voice Attributes:",
        ]
        for name, data in voice.items():
            lines.append(f"- {name}: {data['definition']}")
            lines.extend(f"  - {rule}" for rule in data["actionable_rules"])
        lines.append("")
        lines.append("Negative Constraints:")
        lines.extend(f"- {item}" for item in BX_STYLE_RULES["negative_constraints"])
        lines.append("")
        lines.append("Few-shot Examples:")
        for example in BX_STYLE_RULES["few_shot_examples"]:
            lines.append(f"Type: {example['type']}")
            lines.append(f"Input: {example['input']}")
            lines.append(f"Output: {example['output']}")
        return "\n".join(lines)

    def build_translation_prompt_sections(
        self,
        target_lang: str,
        source_lang: str = "English",
        bx_style_on: bool = False,
        rag_context: str | None = None,
        row_key: str = "",
        glossary_context=None,
        target_lang_code: str = "",
    ) -> list[dict]:
        lang_content = self._build_language_section(target_lang)
        bx_content = self._build_bx_section(target_lang) if bx_style_on else ""
        rag_content = (
            f"{self._normalize_rag_section(rag_context)}\n"
            "Use these examples as style and terminology reference to maintain consistency."
        ) if rag_context else ""

        return [
            {"id": "persona", "label": "PERSONA", "module_key": None, "active": True, "always": True,
             "content": self._build_persona_section(target_lang, source_lang, bx_style_on)},
            {"id": "common", "label": "COMMON LOCALIZATION STANDARD", "module_key": "common", "active": True, "always": True,
             "content": self._build_common_section()},
            {"id": "language", "label": "LANGUAGE SPECIFIC RULE", "module_key": "language",
             "active": bool(lang_content), "always": False,
             "content": lang_content or "(이 타겟 언어에 적용되는 언어별 규칙 없음)"},
            {"id": "bx", "label": "SAMSUNG BX STYLE", "module_key": "bx",
             "active": bool(bx_style_on), "always": False,
             "content": bx_content or "(BX 스타일 비활성화)"},
            {"id": "rag", "label": "RAG TRANSLATION MEMORY", "module_key": "rag",
             "active": bool(rag_content), "always": False,
             "content": rag_content or "(유사 번역 예시 없음 — RAG DB 미연결 또는 유사 항목 없음)"},
            {"id": "formatting", "label": "FORMAT & TYPOGRAPHY RULES", "module_key": "formatting",
             "active": True, "always": True,
             "content": self._build_formatting_section(target_lang, row_key, target_lang_code, bool(glossary_context))},
            {"id": "output", "label": "OUTPUT FORMAT", "module_key": None, "active": True, "always": True,
             "content": 'OUTPUT: Return ONLY a JSON object with a "translation" key.'},
        ]

    def build_audit_prompt_sections(
        self,
        target_lang: str,
        target_lang_code: str | None = None,
        row_key: str = "",
        glossary_context=None,
    ) -> list[dict]:
        lang_content = self._build_language_section(target_lang, korean_heading=True)
        return [
            {"id": "intro", "label": "검수자 역할 정의", "module_key": None, "active": True, "always": True,
             "content": AUDIT_INTRO},
            {"id": "language", "label": "언어별 현지화 기준", "module_key": "language",
             "active": bool(lang_content), "always": False,
             "content": lang_content or "(이 타겟 언어에 적용되는 언어별 규칙 없음)"},
            {"id": "formatting", "label": "FORMAT & TYPOGRAPHY RULES", "module_key": "formatting",
             "active": True, "always": True,
             "content": self._build_formatting_section(target_lang, row_key, target_lang_code or "", bool(glossary_context))},
            {"id": "checklist", "label": "검수 가이드라인", "module_key": None, "active": True, "always": True,
             "content": self._build_audit_checklist()},
            {"id": "glossary_target", "label": "Glossary Target", "module_key": "glossary",
             "active": bool(glossary_context), "always": False,
             "content": f"[Glossary Target]\nTarget code: {target_lang_code or target_lang}" if glossary_context else "(용어집 없음)"},
            {"id": "output_format", "label": "출력 형식", "module_key": None, "active": True, "always": True,
             "content": self._build_audit_output_format()},
        ]

    def _build_audit_checklist(self) -> str:
        lines = ["[검수 가이드라인]"]
        for i, (cat, desc) in enumerate(AUDIT_CHECKLIST_RULES, 1):
            lines.append(f"{i}. {cat}: {desc}")
        return "\n".join(lines)

    def _build_audit_output_format(self) -> str:
        last = len(AUDIT_CHECKLIST_RULES) - 1
        cat_lines = [
            f'    {{"category": "{cat}", "comment": "상세한 분석 결과"}}{"," if i < last else ""}'
            for i, (cat, _) in enumerate(AUDIT_CHECKLIST_RULES)
        ]
        grade_opts = " | ".join(AUDIT_GRADE_CRITERIA)
        grade_criteria = "\n".join(f'  - "{k}": {v}' for k, v in AUDIT_GRADE_CRITERIA.items())
        return (
            "[출력 형식]\nJSON 형식으로 반환하세요:\n"
            "{\n"
            '  "evaluation": [\n'
            + "\n".join(cat_lines)
            + "\n  ],\n"
            f'  "grade": "{grade_opts}",\n'
            '  "suggested_fix": "가장 자연스럽고 정확한 전체 문장 수정안 (수정 불필요 시 빈 문자열)"\n'
            "}\n"
            f"grade 기준:\n{grade_criteria}"
        )

    def _build_formatting_section(
        self,
        target_lang: str,
        row_key: str,
        target_lang_code: str = "",
        glossary_available: bool = True,
    ) -> str:
        lines = [
            "[GLOSSARY RULES]",
        ]

        if glossary_available:
            lines.extend([
                f"- {GLOSSARY_TERM_RULES['rules'][0]}",
                "- Apply term-specific rule or remark exceptions before generic formatting rules.",
            ])
        else:
            lines.append("No glossary terms are provided for this source text.")

        context_mode = self.get_glossary_context_mode(row_key)

        # No-bracket rule is a structural rule for title/button — always add it regardless of glossary.
        if context_mode == "title_button":
            lines.append(f"- {GLOSSARY_NO_BRACKET_INSTRUCTION}")
        elif glossary_available:
            brackets = self.get_brackets(target_lang)
            wrap_rule = GLOSSARY_BRACKET_WRAP_RULE.format(open=brackets[0], close=brackets[1])
            if context_mode == "disclaimer":
                wrap_rule += f" {GLOSSARY_DISCLAIMER_NAV_EXCEPTION}"
            lines.append(f"- {wrap_rule}")

        # Nav path quote rule is typography, not glossary — always applies for disclaimer rows.
        if context_mode == "disclaimer":
            is_ja = target_lang and ("Japanese" in target_lang or "일본" in target_lang)
            if self._is_us_english(target_lang, target_lang_code):
                quote_rule = GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_US
            elif is_ja:
                quote_rule = GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_JA
            elif target_lang_code:
                quote_rule = GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_INTL
            else:
                quote_rule = GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE
            lines.append(f"- {quote_rule}")

        lines += [
            f"\n[{TYPOGRAPHY_AND_PUNCTUATION_RULES['name']}]",
            *[f"- {r}" for r in TYPOGRAPHY_AND_PUNCTUATION_RULES["rules"]],
        ]
        return "\n".join(lines)

    def _normalize_rag_section(self, rag_context: str) -> str:
        rag_section = rag_context.strip()
        if not rag_section.startswith("[Translation Memory Examples]"):
            rag_section = "[Translation Memory Examples]\n" + rag_section
        return rag_section
