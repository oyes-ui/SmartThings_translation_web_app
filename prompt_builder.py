# -*- coding: utf-8 -*-
"""
Prompt builder for translation and audit prompts.

The public methods preserve the existing JSON response contracts while making
prompt modules visible and reusable.
"""

import re

from prompt_modules import (
    BX_STYLE_RULES,
    COMMON_LOCALIZATION_STANDARD,
    GLOBAL_GLOSSARY_STANDARDS,
    LANGUAGE_SPECIFIC_GLOSSARY_RULES,
    GLOSSARY_EXEMPT_MARKERS,
    LANGUAGE_LOCALIZATION_RULES,
)


class PromptBuilder:
    def get_language_rule(self, target_lang: str):
        if not target_lang:
            return None
        normalized = target_lang.lower()
        for lang_key, rule in LANGUAGE_LOCALIZATION_RULES.items():
            if lang_key.lower() in normalized:
                return lang_key, rule
        return None

    def get_brackets(self, target_lang: str) -> str:
        if target_lang and ("Japanese" in target_lang or "일본" in target_lang):
            return "「」"
        return "[]"

    def get_exempt_markers(self) -> list[str]:
        return GLOSSARY_EXEMPT_MARKERS

    def should_skip_brackets(self, row_key: str) -> bool:
        key = (row_key or "").strip().lower()
        if "description" in key or "disclaimer" in key:
            return False
        return "title" in key or "button" in key or bool(re.search(r"\d+$", key))

    def should_wrap_glossary(self, row_key: str, rule_text: str = "") -> bool:
        """Single source of truth for whether a term should be wrapped in brackets."""
        if self.should_skip_brackets(row_key):
            return False
        
        if rule_text:
            clean_rule = rule_text.lower()
            if any(marker in clean_rule for marker in self.get_exempt_markers()):
                return False
                
        return True

    def build_input_formatting(self, target_lang: str) -> dict:
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

        if bx_style_on:
            sections.append(self._build_bx_section(target_lang))

        if rag_context:
            sections.append(
                "[Translation Memory Examples]\n"
                f"{rag_context}\n"
                "Use these examples as style and terminology reference to maintain consistency."
            )

        sections.append(self._build_glossary_section(bool(glossary_context)))
        sections.append(self._build_formatting_section(target_lang, row_key))
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
        sections = [
            "당신은 언어별 문법과 문맥을 정밀하게 분석하는 전문 번역 검수자입니다. 반드시 JSON 형식으로만 응답합니다.",
            self._build_common_section(korean_heading=True),
        ]

        language_section = self._build_language_section(target_lang, korean_heading=True)
        if language_section:
            sections.append(language_section)

        sections.append(self._build_formatting_section(target_lang, row_key))

        sections.append(
            "[검수 가이드라인]\n"
            "1. 문법/유창성: 오타, 문법 오류, 성수 일치, 관용구 사용 등 정밀 점검.\n"
            "2. 정확성: 원문의 뉘앙스와 정보가 정확히 전달되었는지 확인.\n"
            "3. 용어집 준수: 제공된 glossary 데이터와 100% 일치하는지 확인 (대소문자, 띄어쓰기 포함).\n"
            "4. 대소문자(Casing): 해당 언어의 문장형 또는 타이틀형 등 일반 규칙 준수 여부.\n"
            "5. 현지화 품질: 문화, 어투, 문체, 지역 표현이 대상 시장에 자연스러운지 확인.\n"
            "6. 품질 등급 평가: 번역의 품질을 다각도로 분석하여 기술하세요."
        )
        if glossary_context:
            sections.append(f"[Glossary Target]\nTarget code: {target_lang_code or target_lang}")

        sections.append(
            "[출력 형식]\n"
            "JSON 형식으로 반환하세요:\n"
            "{\n"
            '  "evaluation": [\n'
            '    {"category": "문법/유창성", "comment": "상세한 분석 결과"},\n'
            '    {"category": "정확성", "comment": "상세한 분석 결과"},\n'
            '    {"category": "용어집 준수", "comment": "상세한 분석 결과"},\n'
            '    {"category": "대소문자 표기", "comment": "상세한 분석 결과"},\n'
            '    {"category": "현지화 품질", "comment": "상세한 분석 결과"},\n'
            '    {"category": "기타 특이사항", "comment": "상세한 분석 결과"}\n'
            "  ],\n"
            '  "is_excellent": true/false,\n'
            '  "suggested_fix": "가장 자연스럽고 정확한 전체 문장 수정안 (수정 불필요 시 빈 문자열)"\n'
            "}"
        )
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
        language_rule = language_match[1] if language_match else None
        bracket_mode = "skip_for_title_button" if self.should_skip_brackets(row_key) else "wrap_for_description"
        brackets = self.get_brackets(target_lang)

        return {
            "common": {
                "active": True,
                "name": COMMON_LOCALIZATION_STANDARD["name"],
                "description": "Meaning preservation, natural local expression, cultural fit, tone, and risky wording controls.",
            },
            "language": {
                "active": bool(language_rule),
                "name": language_rule["name"] if language_rule else "No language-specific module",
                "description": "; ".join(language_rule["rules"]) if language_rule else "Only the common localization standard is applied.",
            },
            "bx": {
                "active": bool(bx_style_on),
                "name": "Samsung BX Style",
                "description": "Confident Explorer persona and OPEN/BOLD/AUTHENTIC voice rules." if bx_style_on else "BX style is off for this run.",
            },
            "formatting": {
                "active": True,
                "name": GLOBAL_GLOSSARY_STANDARDS["name"],
                "description": f"Bracket mode: {bracket_mode}; brackets: {brackets}.",
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
        _, rule = match
        heading = "[언어별 현지화 기준]" if korean_heading else "[LANGUAGE SPECIFIC RULE]"
        return f"{heading}\n{rule['name']}\n" + "\n".join(f"- {item}" for item in rule["rules"])

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

    def _build_glossary_section(self, glossary_available: bool) -> str:
        if glossary_available:
            return (
                "[GLOSSARY RULE]\n"
                "Use the provided glossary exactly. Apply term-specific rule or remark exceptions before generic formatting rules."
            )
        return "[GLOSSARY RULE]\nNo glossary terms are provided for this source text."

    def _build_formatting_section(self, target_lang: str, row_key: str) -> str:
        formatting_rules = []

        # 1. Global standards
        formatting_rules.append(f"[{GLOBAL_GLOSSARY_STANDARDS['name']}]")
        formatting_rules.extend(f"- {r}" for r in GLOBAL_GLOSSARY_STANDARDS["rules"])

        # 2. Bracket style (Default vs Japanese)
        from prompt_modules import (
            DEFAULT_GLOSSARY_BRACKET_RULE,
            LANGUAGE_SPECIFIC_GLOSSARY_RULES
        )

        brackets = ("[", "]")
        if target_lang == "Japanese":
            ja_rules = LANGUAGE_SPECIFIC_GLOSSARY_RULES.get("Japanese", {})
            formatting_rules.append(f"\n[{ja_rules.get('name', 'Japanese Bracket Style')}]")
            formatting_rules.extend(f"- {r}" for r in ja_rules.get("rules", []))
            brackets = ("「", "」")
        else:
            formatting_rules.append(f"\n[{DEFAULT_GLOSSARY_BRACKET_RULE['name']}]")
            formatting_rules.extend(f"- {r}" for r in DEFAULT_GLOSSARY_BRACKET_RULE["rules"])
            brackets = ("[", "]")

        # 3. Dynamic context-based rule
        if self.should_skip_brackets(row_key):
            context_rule = f"Use glossary terms WITHOUT brackets for this {row_key} context."
        else:
            context_rule = (
                f"Wrap glossary terms in '{brackets[0]}' and '{brackets[1]}' for Description context unless an exception applies."
            )
        
        formatting_rules.append(f"- {context_rule}")

        return (
            "[FORMAT AND GLOSSARY RULES]\n"
            + "\n".join(formatting_rules)
        )
