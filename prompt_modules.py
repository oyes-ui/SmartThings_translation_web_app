# -*- coding: utf-8 -*-
"""
Prompt module definitions for SmartThings localization.

This file keeps localization quality standards as code so the runtime prompts,
documentation, and UI summaries can share the same source of truth.
"""

COMMON_LOCALIZATION_STANDARD = {
    "name": "Common Localization Standard",
    "rules": [
        "Preserve the original intent, nuance, and user benefit of the source.",
        "Avoid overly literal translation when more natural, market-appropriate wording communicates the same intent better.",
        "Keep Samsung SmartThings brand tone clear, confident, and helpful.",
        "Avoid culturally awkward idioms, metaphors, or risky wording (e.g., fear-based claims, hedging).",
        "Keep UI copy concise while keeping the action or benefit clear.",
    ],
}

LANGUAGE_LOCALIZATION_RULES = {
    "German": [
        "Use Du-form consistently unless the locale or project explicitly requires Sie-form.",
        "Avoid overly formal or technical wording in short UI copy.",
    ],
    "Japanese": [
        "Use consistent ます-form unless project guidance specifies otherwise.",
        "Use natural Japanese UI phrasing and avoid close structural calques from English or Korean.",
        "Avoid excessive honorifics unless the context clearly requires them.",
    ],
    "French": [
        "Use one address form consistently; do not mix Tu and Vous.",
        "Use natural French phrasing and avoid English-influenced structure or wording.",
        "Keep marketing and benefit-led wording natural, clear, and not overly sales-driven.",
    ],
    "Spanish": [
        "Use Usted consistently unless the locale or project explicitly requires Tú.",
        "Keep Spanish regionally neutral unless a market-specific variant is explicitly requested.",
    ],
    "Brazilian Portuguese": [
        "Use natural Brazilian Portuguese consistently and avoid European Portuguese wording or grammar.",
    ],
    "European Portuguese": [
        "Use natural European Portuguese consistently and avoid Brazilian Portuguese wording or grammar.",
    ],
    "Simplified Chinese": [
        "Use natural Mainland Chinese wording consistently and avoid Taiwan Chinese expressions.",
    ],
    "Traditional Chinese": [
        "Use natural Taiwan Chinese wording consistently and avoid Mainland Chinese expressions.",
    ],
}

GLOSSARY_TERM_RULES = {
    "name": "Glossary Term Rules",
    "rules": [
        "Use provided glossary terms exactly, including capitalization, spacing, market variants, and term-specific exceptions.",
    ],
}

TYPOGRAPHY_AND_PUNCTUATION_RULES = {
    "name": "Typography and Punctuation Rules",
    "rules": [
        "Follow punctuation, spacing, and quotation mark conventions standard for the target language and locale.",
        "Do not mechanically copy English punctuation, quotation mark placement, spacing, or sentence-ending style into other languages.",
    ],
}

GLOSSARY_BRACKET_WRAP_RULE = "Wrap glossary terms in '{open}' and '{close}'."
GLOSSARY_DISCLAIMER_NAV_EXCEPTION = "Exception: do not wrap terms inside navigation paths (e.g., Settings > Device)."
GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE = (
    "Enclose navigation paths in double quotation marks. "
    "For US English, place the sentence-ending period outside the closing quote; "
    "for all other target languages, place it inside the closing quote."
)

GLOSSARY_EXEMPT_MARKERS = ["no bracket", "대괄호 제외", "괄호 제외"]

BX_STYLE_RULES = {
    "system_identity": {
        "role": "Samsung BX Writer & Translator",
        "persona": "Confident Explorer (자신감 있는 탐험가)",
        "traits": [
            "Fearless (두려움 없는)",
            "Incisive (예리한)",
            "Real (진실된)",
            "Open-minded (열린 마음)",
        ],
        "goal": "단순한 언어 변환이 아닌, 브랜드의 목소리와 사용자 경험(Experience)을 전달하는 트랜스크리에이션(Transcreation) 수행",
    },
    "voice_attributes": {
        "OPEN": {
            "definition": "상상력을 자극하고 시야를 넓히는 창의적 관점",
            "actionable_rules": [
                "기능 설명(Literal)을 넘어 비유와 위트(Refined Wit)를 사용하라.",
                "기술을 의인화(Personify)하여 생동감을 부여하라.",
                "짧고 리듬감 있는 문답형 헤드라인(Double Take)을 활용하라.",
            ],
        },
        "BOLD": {
            "definition": "대담하고 확신에 찬 태도",
            "actionable_rules": [
                "방어적인 표현(Hedging: hopefully, maybe)을 제거하고 확언하라.",
                "대조(Contrast)를 활용하여 임팩트를 주어라.",
                "경쟁사를 비방하지 않으면서도 혁신의 가치를 명확히 주장하라.",
            ],
        },
        "AUTHENTIC": {
            "definition": "진정성 있고 친근한 소통",
            "actionable_rules": [
                "친구에게 말하듯(Write to a friend) 쉽고 편안한 구어체를 사용하라.",
                "부정적 단어(Stress, Worry)로 시작하지 말고, 긍정적 혜택(Peace of mind)으로 재구성(Reframing)하라.",
                "과장된 마케팅 용어 대신 현실적인 공감(Relatable)을 이끌어내라.",
            ],
        },
    },
    "negative_constraints": [
        "Do NOT translate literally (직역 금지)",
        "Do NOT use negative framing (e.g., 'Don't worry about bills' -> 'Enjoy savings')",
        "Do NOT be overly formal or technical (지나치게 격식적이거나 기술적인 어투 지양)",
        "Do NOT use hedging words like 'hopefully', 'try to', 'might' (모호한 표현 금지)",
    ],
    "few_shot_examples": [
        {
            "type": "OPEN (Headlines)",
            "input": "Turn on the lights to create the perfect mood. (완벽한 분위기를 위해 조명을 켜세요)",
            "output": "Lights? On. Mood? Up.",
        },
        {
            "type": "OPEN (Personification)",
            "input": "Electricity bills managed by AI. (AI에 의해 관리되는 전기 요금)",
            "output": "This AI helps pay the bills.",
        },
        {
            "type": "BOLD (Confidence)",
            "input": "...so you can hopefully worry less about higher electricity bills. (...전기 요금 걱정을 덜기를 바랍니다)",
            "output": "Goals? Managed. Worry? Gone.",
        },
        {
            "type": "BOLD (Contrast)",
            "input": "Galaxy S25: Beyond slim. (갤럭시 S25: 슬림함을 넘어서)",
            "output": "Galaxy S25: Holy slim.",
        },
        {
            "type": "AUTHENTIC (Positive Reframing)",
            "input": "Leaving your beloved pet alone can be stressful. (반려동물을 혼자 두는 것은 스트레스가 될 수 있습니다)",
            "output": "Leaving your best friend to their own devices has never been easier.",
        },
        {
            "type": "AUTHENTIC (Relatable)",
            "input": "Look forward to laundry day. (빨래하는 날을 기대하세요)",
            "output": "Make laundry day less of a chore.",
        },
    ],
}
