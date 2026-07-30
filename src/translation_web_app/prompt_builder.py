# -*- coding: utf-8 -*-
"""
Prompt builder for translation and audit prompts.

The public methods preserve the existing JSON response contracts while making
prompt modules visible and reusable.
"""

import json
import re

from translation_web_app.prompt_modules import (
    GLOSSARY_DEACTIVATION_MARKERS,
    GLOSSARY_EXEMPT_MARKERS,
    resolve_language_identifier,
)
from translation_web_app.rules_loader import get_rules

# Loaded at import so a malformed rule file stops the app from starting rather
# than silently producing prompts with missing rules.
_RULES = get_rules()


class PromptBuilder:
    def get_language_rule(self, target_lang: str):
        if not target_lang:
            return None
        normalized = resolve_language_identifier(target_lang).lower()
        languages = _RULES.languages
        for lang_key in languages:
            if lang_key.lower() == normalized:
                return lang_key, list(languages[lang_key].prompt_rules())
        # Longest key first so "english_us..." matches English_US, not English.
        # The secondary key makes equal-length ties independent of file order.
        for lang_key in sorted(languages, key=lambda key: (-len(key), key)):
            if lang_key.lower() in normalized:
                return lang_key, list(languages[lang_key].prompt_rules())
        return None

    def get_brackets(self, target_lang: str) -> str:
        if target_lang and ("Japanese" in target_lang or "일본" in target_lang):
            return "「」"
        return "[]"

    def get_exempt_markers(self) -> list[str]:
        return GLOSSARY_EXEMPT_MARKERS

    def is_glossary_deactivated(self, rule_text: str = "") -> bool:
        """Term-level deactivation: the term is not glossary-enforced at all."""
        clean_rule = (rule_text or "").lower().replace(" ", "")
        return any(marker in clean_rule for marker in GLOSSARY_DEACTIVATION_MARKERS)

    def has_exempt_marker(self, rule_text: str = "") -> bool:
        """Term-level bracket exemption (e.g. '대괄호 제외')."""
        clean_rule = (rule_text or "").lower()
        return any(marker in clean_rule for marker in self.get_exempt_markers())

    def get_glossary_context_mode(self, row_key: str) -> str:
        key = (row_key or "").strip().lower()
        if "disclaimer" in key:
            return "disclaimer"
        if "description" in key:
            return "description"
        if "title" in key or "button" in key or bool(re.search(r"\d+$", key)):
            return "title_button"
        return "description"

    def should_skip_brackets(self, row_key: str) -> bool:
        return self.get_glossary_context_mode(row_key) == "title_button"

    def resolve_glossary_bracket_policy(
        self,
        row_key: str = "",
        rule_text: str = "",
        inside_nav_path: bool = False,
    ) -> str:
        """Single source of truth for a glossary term's bracket policy.

        Returns one of ``"skip"`` | ``"no_bracket"`` | ``"wrap"``. Both the prompt
        assembly and the deterministic checker/normalizer consume this so their
        precedence can never diverge.

        Priority (high → low):
          1. deactivated term (rule marks it off)          → ``skip``
          2. structural title/heading/button context       → ``no_bracket``
          3. occurrence sits inside a navigation path       → ``no_bracket``
          4. term-specific bracket exemption (대괄호 제외)  → ``no_bracket``
          5. default                                        → ``wrap``

        ``inside_nav_path`` is an OCCURRENCE-level fact: the same term may resolve
        to ``no_bracket`` inside a path and ``wrap`` outside it within one cell.
        """
        if self.is_glossary_deactivated(rule_text):
            return "skip"
        if self.should_skip_brackets(row_key):
            return "no_bracket"
        if inside_nav_path:
            return "no_bracket"
        if self.has_exempt_marker(rule_text):
            return "no_bracket"
        return "wrap"

    def should_wrap_glossary(self, row_key: str, rule_text: str = "") -> bool:
        """Thin wrapper over :meth:`resolve_glossary_bracket_policy` (cell-level,
        i.e. not inside a navigation path). Kept for backward compatibility."""
        return self.resolve_glossary_bracket_policy(
            row_key=row_key, rule_text=rule_text, inside_nav_path=False
        ) == "wrap"

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
    ) -> str:
        sections = []
        sections.append(self._build_persona_section(target_lang, source_lang, bx_style_on))
        sections.append(self._build_common_section())

        language_section = self._build_language_section(target_lang)
        if language_section:
            sections.append(language_section)

        if bx_style_on or self._is_bx_lang(target_lang):
            sections.append(self._build_bx_section(target_lang))

        if rag_context:
            rag_section = self._normalize_rag_section(rag_context)
            sections.append(
                f"{rag_section}\n"
                "Use these examples as style and terminology reference to maintain consistency."
            )

        sections.append(self._build_formatting_section(target_lang, row_key, bool(glossary_context)))
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
        sections = [_RULES.doc("audit").one("intro")]

        language_section = self._build_language_section(target_lang, korean_heading=True)
        if language_section:
            sections.append(language_section)

        sections.append(self._build_formatting_section(target_lang, row_key, bool(glossary_context)))
        sections.append(self._build_audit_checklist())

        if glossary_context:
            sections.append(f"[Glossary Target]\nTarget code: {target_lang_code or target_lang}")

        sections.append(self._build_audit_output_format())
        return "\n\n".join(sections).strip()

    def build_bx_audit_prompt(self, source_text: str, translated_text: str, target_lang: str) -> str:
        identity = _RULES.bx
        return f"""You are a Samsung BX Audit Expert.
Evaluate if the following translation aligns with the Samsung BX Persona and Voice Attributes.

Source: {source_text}
Translation: {translated_text}
Target Language: {target_lang}

Persona: {identity.persona}
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
                "name": _RULES.doc("common").display_name,
                "description": "Meaning preservation, natural local expression, cultural fit, tone, and risky wording controls.",
            },
            "language": {
                "active": bool(language_rules),
                "name": _RULES.languages[lang_key].display_name if lang_key else "No language-specific module",
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
                "name": _RULES.doc("typography").display_name,
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
        if bx_style_on or self._is_bx_lang(target_lang):
            role = _RULES.bx.role
            return (
                f"You are the {role}.\n"
                f"Source Language: {source_lang}\n"
                f"Target Language: {target_lang}\n"
                "TASK: Translate and polish the source text naturally for a native speaker while faithfully conveying the full intent and nuance of the original."
            )
        return (
            f"You are a professional {target_lang} localizer.\n"
            f"Source Language: {source_lang}\n"
            f"Target Language: {target_lang}\n"
            "TASK: Translate the source text naturally for a native speaker while faithfully conveying the full intent and nuance of the original."
        )

    def _build_common_section(self, korean_heading: bool = False) -> str:
        heading = "[공통 현지화 품질 기준]" if korean_heading else "[COMMON LOCALIZATION STANDARD]"
        texts = _RULES.doc("common").texts("standard")
        return heading + "\n" + "\n".join(f"- {text}" for text in texts)

    def _build_language_section(self, target_lang: str, korean_heading: bool = False) -> str:
        match = self.get_language_rule(target_lang)
        if not match:
            return ""
        lang_key, rules = match
        if not rules:
            # A file may carry only agent_audit rules; emit no headed empty section.
            return ""
        heading = "[언어별 현지화 기준]" if korean_heading else "[LANGUAGE SPECIFIC RULE]"
        label = _RULES.languages[lang_key].display_name
        return f"{heading}\n{label}\n" + "\n".join(f"- {item}" for item in rules)

    _BX_AUTO_LANGS = {"English_US", "English"}

    def _is_bx_lang(self, target_lang: str) -> bool:
        return target_lang in self._BX_AUTO_LANGS

    def _build_bx_section(self, target_lang: str) -> str:
        bx = _RULES.bx
        lines = [
            "[SAMSUNG BX STYLE]",
            f"Persona: {bx.persona}",
            f"Goal: {bx.goal}",
            f"Target Language: {target_lang}",
            "",
            "Voice Attributes:",
        ]
        for name, actionable_rules in bx.voice_attributes.items():
            lines.append(f"- {name}:")
            lines.extend(f"  - {rule}" for rule in actionable_rules)
        lines.append("")
        lines.append("Negative Constraints:")
        lines.extend(f"- {item}" for item in bx.negative_constraints)
        lines.append("")
        lines.append("Few-shot Examples:")
        for example in bx.few_shot_examples:
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
        glossary_available: bool = False,
    ) -> list[dict]:
        lang_content = self._build_language_section(target_lang)
        bx_content = self._build_bx_section(target_lang) if (bx_style_on or self._is_bx_lang(target_lang)) else ""
        rag_content = (
            f"{self._normalize_rag_section(rag_context)}\n"
            "Use these examples as style and terminology reference to maintain consistency."
        ) if rag_context else ""

        has_glossary = glossary_available or bool(glossary_context)

        return [
            {"id": "persona", "label": "PERSONA", "module_key": None, "active": True, "always": True,
             "content": self._build_persona_section(target_lang, source_lang, bx_style_on)},
            {"id": "common", "label": "COMMON LOCALIZATION STANDARD", "module_key": "common", "active": True, "always": True,
             "content": self._build_common_section()},
            {"id": "language", "label": "LANGUAGE SPECIFIC RULE", "module_key": "language",
             "active": bool(lang_content), "always": False,
             "content": lang_content or "(이 타겟 언어에 적용되는 언어별 규칙 없음)"},
            {"id": "bx", "label": "SAMSUNG BX STYLE", "module_key": "bx",
             "active": bool(bx_style_on or self._is_bx_lang(target_lang)), "always": False,
             "content": bx_content or "(BX 스타일 비활성화)"},
            {"id": "rag", "label": "RAG TRANSLATION MEMORY", "module_key": "rag",
             "active": bool(rag_content), "always": False,
             "content": rag_content or "(유사 번역 예시 없음 — RAG DB 미연결 또는 유사 항목 없음)"},
            {"id": "formatting", "label": "FORMAT & TYPOGRAPHY RULES", "module_key": "formatting",
             "active": True, "always": True,
             "content": self._build_formatting_section(target_lang, row_key, has_glossary)},
            {"id": "output", "label": "OUTPUT FORMAT", "module_key": None, "active": True, "always": True,
             "content": 'OUTPUT: Return ONLY a JSON object with a "translation" key.'},
        ]

    def build_audit_prompt_sections(
        self,
        target_lang: str,
        target_lang_code: str | None = None,
        row_key: str = "",
        glossary_context=None,
        glossary_available: bool = False,
    ) -> list[dict]:
        lang_content = self._build_language_section(target_lang, korean_heading=True)
        has_glossary = glossary_available or bool(glossary_context)

        if has_glossary:
            if glossary_context:
                terms_str = "\n".join(f"  - {k} -> {v}" for k, v in glossary_context.items())
                glossary_content = f"[Glossary Target]\nTarget code: {target_lang_code or target_lang}\nMatched terms:\n{terms_str}"
            else:
                glossary_content = f"[Glossary Target]\nTarget code: {target_lang_code or target_lang}\n(No matching glossary terms found for this source text)"
        else:
            glossary_content = "(용어집 없음)"

        return [
            {"id": "intro", "label": "검수자 역할 정의", "module_key": None, "active": True, "always": True,
             "content": _RULES.doc("audit").one("intro")},
            {"id": "language", "label": "언어별 현지화 기준", "module_key": "language",
             "active": bool(lang_content), "always": False,
             "content": lang_content or "(이 타겟 언어에 적용되는 언어별 규칙 없음)"},
            {"id": "formatting", "label": "FORMAT & TYPOGRAPHY RULES", "module_key": "formatting",
             "active": True, "always": True,
             "content": self._build_formatting_section(target_lang, row_key, has_glossary)},
            {"id": "checklist", "label": "검수 가이드라인", "module_key": None, "active": True, "always": True,
             "content": self._build_audit_checklist()},
            {"id": "glossary_target", "label": "Glossary Target", "module_key": "glossary",
             "active": has_glossary, "always": False,
             "content": glossary_content},
            {"id": "output_format", "label": "출력 형식", "module_key": None, "active": True, "always": True,
             "content": self._build_audit_output_format()},
        ]

    def _build_audit_checklist(self) -> str:
        lines = ["[검수 가이드라인]"]
        for i, (cat, desc) in enumerate(_RULES.doc("audit").labelled("checklist"), 1):
            lines.append(f"{i}. {cat}: {desc}")
        return "\n".join(lines)

    def _build_audit_output_format(self) -> str:
        checklist = _RULES.doc("audit").labelled("checklist")
        grades = _RULES.doc("audit").labelled("grade")
        last = len(checklist) - 1
        cat_lines = [
            f'    {{"category": "{cat}", "comment": "상세한 분석 결과"}}{"," if i < last else ""}'
            for i, (cat, _) in enumerate(checklist)
        ]
        grade_opts = " | ".join(label for label, _ in grades)
        grade_criteria = "\n".join(f'  - "{k}": {v}' for k, v in grades)
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
        glossary_available: bool = True,
    ) -> str:
        lines = [
            "[GLOSSARY RULES]",
        ]

        glossary_doc = _RULES.doc("glossary")
        typography_doc = _RULES.doc("typography")

        if glossary_available:
            lines.extend([
                f"- {glossary_doc.one('term_rule')}",
                f"- {glossary_doc.one('bracket_precedence')}",
            ])
        else:
            lines.append("No glossary terms are provided for this source text.")

        context_mode = self.get_glossary_context_mode(row_key)

        # No-bracket rule is a structural rule for title/button — always add it regardless of glossary.
        if context_mode == "title_button":
            lines.append(f"- {glossary_doc.one('no_bracket')}")
        elif glossary_available:
            brackets = self.get_brackets(target_lang)
            wrap_rule = glossary_doc.one("bracket_wrap").format(
                open=brackets[0], close=brackets[1]
            )
            if context_mode == "disclaimer":
                wrap_rule += f" {glossary_doc.one('nav_exception')}"
            lines.append(f"- {wrap_rule}")

        # Nav path quote rule is typography, not glossary — always applies for disclaimer rows.
        if context_mode == "disclaimer":
            is_east_asian = target_lang and any(k in target_lang for k in (
                "Japanese", "일본", "Chinese", "중국", "Taiwan", "대만"
            ))
            slot = "nav_quote_east_asian" if is_east_asian else "nav_quote_default"
            lines.append(f"- {glossary_doc.one(slot)}")

        lines += [
            f"\n[{typography_doc.display_name}]",
            *[f"- {text}" for text in typography_doc.texts("rule")],
        ]
        return "\n".join(lines)

    def _normalize_rag_section(self, rag_context: str) -> str:
        rag_section = rag_context.strip()
        if not rag_section.startswith("[Translation Memory Examples]"):
            rag_section = "[Translation Memory Examples]\n" + rag_section
        return rag_section
