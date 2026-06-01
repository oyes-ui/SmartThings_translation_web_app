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
        "Where appropriate, use creative expression — refined wit, metaphor, or personification — to make technology feel approachable rather than technical.",
        "Avoid culturally awkward idioms, fear-based framing, overly formal or technical language, and hedging words (e.g., 'hopefully', 'try to', 'might').",
        "Keep UI copy concise while keeping the action or benefit clear.",
    ],
}

LANGUAGE_LOCALIZATION_RULES = {
    "Korean": [
        "Use consistent polite (honorific) or formal literary style as appropriate for the context.",
        "Ensure terminology consistency throughout the UI.",
    ],
    "English_US": [
        "Use US English spelling and wording (e.g., 'color', 'personalize').",
        "Follow the project-specific rule for disclaimers: place the sentence-ending period outside the closing quotation mark.",
    ],
    "English_UK": [
        "Use British English spelling (e.g., 'colour', 'personalise', 'optimise').",
        "Avoid US-specific vocabulary and phrasing.",
    ],
    "English_AU": [
        "Use Australian English with British-style spelling.",
        "Avoid overly aggressive US-style marketing tones; keep it helpful and clear.",
    ],
    "English_SG": [
        "Use concise, international Singapore English with British-style spelling where appropriate.",
        "Do NOT use Singlish, local slang, or overly US-specific wording.",
    ],
    "English": [
        "Use US English spelling and wording (e.g., 'color', 'personalize') unless a more specific English market variant is specified.",
        "Follow the project-specific rule for disclaimers: place the sentence-ending period outside the closing quotation mark.",
    ],
    "German": [
        "Use Du-form consistently unless the locale or project explicitly requires Sie-form.",
        "Ensure natural capitalization of nouns and maintain natural compound word structures.",
        "Avoid overly formal or technical wording in short UI copy.",
        "Use „...“ (German quotation marks) for quoted text and navigation paths; do not use English straight quotation marks.",
    ],
    "Japanese": [
        "Use consistent ます-form unless project guidance specifies otherwise.",
        "Use natural Japanese UI phrasing and avoid close structural calques from English or Korean.",
        "Prioritize natural '操作/設定' style expressions over mechanical literal translations.",
        "Avoid excessive honorifics unless the context clearly requires them.",
    ],
    "French": [
        "Use Vous-form consistently; do not use Tu-form.",
        "Use «...» (guillemets) for quoted text and navigation paths.",
        "Avoid unnecessary capitalization in UI copy.",
        "Use natural French phrasing and avoid English-influenced structures.",
    ],
    "French_Belgium": [
        "Use Vous-form consistently; do not use Tu-form.",
        "Use «...» (guillemets) for quoted text and navigation paths.",
        "Use neutral French and avoid overly idiomatic expressions specific to mainland France.",
        "Ensure consistent tone for the Belgian market.",
    ],
    "French_Canada": [
        "Use Vous-form consistently; do not use Tu-form.",
        "Use «...» (guillemets) for quoted text and navigation paths.",
        "Follow Canadian French standards; prioritize phrasing natural to North American French over mainland France idioms.",
    ],
    "Italian": [
        "Use natural Italian UI sentence structures; avoid English-style noun-chaining.",
    ],
    "Spanish": [
        "Use tú (informal address) consistently; do not use Usted unless the source or project explicitly requires formal register.",
        "Use Spain Spanish (Castilian) and avoid Latin American-specific wording or usage.",
    ],
    "Dutch": [
        "Use 'u/uw' (formal address) consistently; do not use je/jij.",
        "Keep Dutch copy direct and concise.",
        "Avoid literal translation of English word order or noun-phrase structures.",
    ],
    "Swedish": [
        "Keep Swedish UI copy concise and natural.",
        "Avoid English-style title case; prioritize sentence case for headings and buttons.",
    ],
    "Arabic": [
        "Use Modern Standard Arabic (MSA).",
        "Follow Arabic conventions for UI directionality, punctuation, and sentence-ending styles.",
        "Avoid literal translations that sound unnatural in Arabic.",
    ],
    "European Portuguese": [
        "Use European Portuguese consistently; avoid Brazilian Portuguese wording, vocabulary, or gerund structures.",
    ],
    "Brazilian Portuguese": [
        "Use Brazilian Portuguese consistently; avoid European Portuguese vocabulary or phrasing.",
    ],
    "Russian": [
        "Use natural Russian word order.",
        "Avoid English-style noun-phrase literal translations and excessive capitalization.",
        "Use «...» (guillemets) for quoted text and navigation paths; do not use English straight quotation marks.",
    ],
    "Turkish": [
        "Use natural Turkish word order and avoid structural calques from English.",
        "Maintain concise imperative or descriptive forms suitable for UI copy.",
    ],
    "Simplified Chinese": [
        "Use natural Mainland Chinese wording and Simplified Chinese characters.",
        "Avoid Taiwan-specific terminology.",
        "Use fullwidth punctuation marks (，。！？) consistently; avoid half-width ASCII punctuation.",
        "Use fullwidth double quotation marks (“...”) for quoted text and navigation paths; do not use 「」 (Japanese-style brackets).",
    ],
    "Traditional Chinese": [
        "Use natural Taiwan Traditional Chinese wording and characters.",
        "Avoid Mainland Chinese-specific terminology.",
        "Use fullwidth punctuation marks (，。！？) consistently; avoid half-width ASCII punctuation.",
        "Use 「...」 for quoted text and navigation paths.",
    ],
    "Polish": [
        "Ensure natural Polish declension and grammatical agreement.",
        "Avoid English-style noun-chaining; use natural phrasing.",
    ],
    "Vietnamese": [
        "Use natural Vietnamese word order and concise UI phrasing.",
        "Avoid English-style capitalization patterns.",
    ],
    "Thai": [
        "Use natural Thai UI phrasing.",
        "Avoid unnecessary spaces and mechanical copying of English punctuation.",
    ],
    "Indonesian": [
        "Keep Indonesian copy concise and natural.",
        "Avoid overly formal structures or English-style literal phrasing.",
    ],
}

GLOSSARY_TERM_RULES = {
    "name": "Glossary Term Rules",
    "rules": [
        "Use provided glossary terms exactly as given — including capitalization, spacing, and market variants. Glossary capitalization is authoritative and overrides title case, sentence case, and heading/button capitalization rules; do not adapt glossary terms for naturalness.",
    ],
}

TYPOGRAPHY_AND_PUNCTUATION_RULES = {
    "name": "Typography and Punctuation Rules",
    "rules": [
        "For quotation marks, spacing, and punctuation specific to the target language, follow the rules stated in the language section above; where no specific rule is given, apply the standard convention for that language and locale.",
        "Do not mechanically copy English punctuation, quotation mark placement, spacing, or sentence-ending style into other languages.",
    ],
}

GLOSSARY_BRACKET_WRAP_RULE = "Wrap glossary terms in '{open}' and '{close}'."
GLOSSARY_DISCLAIMER_NAV_EXCEPTION = "Exception: do not wrap terms inside navigation paths (e.g., Settings > Device)."
GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE = (
    "Enclose navigation paths in quotation marks appropriate for the target language. "
    "Place the sentence-ending period outside the closing quotation mark."
)

GLOSSARY_EXEMPT_MARKERS = ["no bracket", "대괄호 제외", "괄호 제외"]

GLOSSARY_NO_BRACKET_INSTRUCTION = (
    "For title, section heading, and button copy, do not wrap glossary terms in any brackets. "
    "Use the glossary term text exactly as provided, without [], 「」, or any other surrounding bracket marks, "
    "even if the source text contains brackets. Do not change glossary capitalization to satisfy heading or button case style."
)
GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_EAST_ASIAN = (
    "Enclose navigation paths in quotation marks appropriate for the target language."
)

AUDIT_INTRO = (
    "당신은 Samsung SmartThings UI 현지화 전문 검수자입니다.\n"
    "최우선 기준: '현지인이 실제로 쓸 법한 자연스러운 표현인가'(현지화). "
    "원문 의미가 보존되었는지는 별도 항목으로 검토하며, "
    "두 기준이 충돌하면 현지화 자연스러움을 우선합니다. "
    "반드시 JSON 형식으로만 응답합니다."
)

AUDIT_CHECKLIST_RULES = [
    (
        "문법/유창성",
        "오타, 문법 오류, 성수 일치, 관용구 사용 등 정밀 점검.",
    ),
    (
        "원문의미 충실도",
        "원문의 핵심 의미·뉘앙스·사용자 혜택이 번역에서 손실 없이 전달되었는지 확인. "
        "직역 여부와 무관하게 '정보 손실' 또는 '의미 왜곡'이 발생했는지만 판단한다.",
    ),
    (
        "용어집 준수",
        "제공된 glossary 데이터와 100% 일치하는지 확인 (대소문자, 띄어쓰기 포함). "
        "항목별 예외 규칙(rule/remark)이 있는 경우 예외가 우선 적용되었는지 확인. "
        "현지화 자연스러움 여부와 관계없이 절대 적용되는 기준이다.",
    ),
    (
        "현지화",
        "해당 언어권 현지인이 실제로 사용하는 자연스러운 표현인지 종합 평가. "
        "① [언어별 현지화 기준] 규칙 준수 (예: 독일어 Du-form, 일본어 ます형, 프랑스어 Tu/Vous 등), "
        "② 직역·구조적 번역이 아닌 시장 맥락에 맞는 표현 선택, "
        "③ 문화적 뉘앙스와 브랜드 보이스(Confident Explorer)의 현지 적용.",
    ),
    (
        "대소문자 표기",
        "대상 언어의 문장형(sentence case) 또는 타이틀형(title case) 등 일반 대소문자 표기 규칙 준수 여부.",
    ),
    (
        "서식 및 표기",
        "[서식 규칙] 섹션 기준으로 점검: glossary 용어의 bracket 표기 적용 여부, "
        "탐색 경로(nav path)의 따옴표 및 마침표 위치, 타이포그래피·구두점·간격 등 대상 언어 표기 규칙 준수 여부.",
    ),
]

AUDIT_GRADE_CRITERIA = {
    "Excellent": "의미 손실 없이 현지인이 자연스럽게 받아들일 표현으로 구현됨. 용어집·서식 완벽 준수.",
    "Good": "의미 보존 및 현지화 방향은 맞으나, 더 자연스러운 표현으로 개선 가능한 부분 존재. 출시 가능 수준.",
    "Needs Revision": "현지화 부자연스러움(직역·어색한 표현), 의미 왜곡, 용어집 불일치, 문법 오류 중 하나 이상 해당.",
}

BX_STYLE_RULES = {
    "system_identity": {
        "role": "Samsung BX Writer & Translator",
        "persona": "Confident Explorer (자신감 있는 탐험가)",
        "goal": (
            "Craft English copy that sounds like a confident, friendly guide — not a tech manual. "
            "Apply OPEN, BOLD, AUTHENTIC voice through specific techniques below."
        ),
    },
    "voice_attributes": {
        "OPEN": {
            "actionable_rules": [
                "Go beyond the literal benefit to reveal all the hidden dimensions — "
                "the experience, emotion, or new perspective behind the feature.",
                "Personify our tech to create intentional wit linked to product functionality. "
                "(e.g., 'This AI helps pay the bills')",
                "Upend expectations: set up a sentence one way, then give it an unexpected ending. "
                "(e.g., 'Visuals so real, real life looks fake.')",
            ],
        },
        "BOLD": {
            "actionable_rules": [
                "Pair technical detail with the real emotional reaction it inspires. "
                "(e.g., 'Reaction: woah.')",
                "Play up contrast to create dramatic effect that underscores the different angles "
                "of our innovation. (e.g., 'Super small. Supremely smart.')",
                "Share our POV: take a clear stance rather than sitting on the fence.",
            ],
        },
        "AUTHENTIC": {
            "actionable_rules": [
                "Write to a friend: imagine writing to someone you know. "
                "Replace technical language with everyday language.",
                "Find the upside: reframe negatives to positives to make our tech feel approachable. "
                "(e.g., 'Look forward to laundry day.')",
                "Find a tangible benefit: pull out a specific, relatable benefit instead of a broad claim. "
                "(e.g., 'Never run out of eggs again.')",
            ],
        },
    },
    "negative_constraints": [
        "Do NOT use negative framing — always reframe into a positive benefit. "
        "(e.g., 'Don't worry about bills' → 'Enjoy savings')",
        "Never try too hard to relate — we're still a premium brand. "
        "Avoid slang or overly casual phrasing.",
        "Never be overly metaphorical — write with purpose and refinement.",
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
