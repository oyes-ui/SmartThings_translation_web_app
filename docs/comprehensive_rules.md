# [가이드 02] 종합 규칙 모음 (Comprehensive Rules)

이 문서는 시스템 내부(`src/translation_web_app/prompt_modules.py`)에 정의된 모든 번역 및 검수 규칙을 한곳에 모은 참조 문서입니다. 소스 코드 내부의 실제 변수명 및 상수를 표기하여 코드와 정책 간 완벽한 동기화(Source of Truth)를 제공합니다.

---

## 1. 번역 관련 (Translation Rules)

### 1.1 공통 현지화 표준 (Common Standards)
모든 언어에 공통적으로 적용되는 기본 원칙입니다.
> **📌 매핑 변수**: `COMMON_LOCALIZATION_STANDARD` (`src/translation_web_app/prompt_modules.py`)

- 원문의 의도, 뉘앙스 및 사용자 혜택을 보존할 것.
- 직역보다는 해당 시장에 적합하고 자연스러운 표현을 우선할 것.
- 적절한 경우 세련된 위트, 비유, 의인화 등 창의적 표현을 사용하여 기술이 전문적이기보다 친근하게 느껴지도록 할 것.
- 문화적으로 어색한 관용구, 공포 유발 프레이밍, 지나치게 격식적이거나 기술적인 어투, 모호한 확신을 주는 단어('hopefully', 'try to', 'might' 등)를 피할 것.
- UI 문구는 간결하게 유지하되, 작업이나 혜택은 명확히 전달할 것.

---

### 1.2 삼성 BX 스타일 가이드 (Samsung BX Style)
`target_lang`이 `English_US` 또는 `English`일 때 자동으로 적용되는 영문 브랜드 보이스 규칙입니다. (구 UI 전역 토글 제거됨)
> **📌 매핑 변수**: `BX_STYLE_RULES` (`src/translation_web_app/prompt_modules.py`)
> **📌 활성화 조건**: `target_lang ∈ {"English_US", "English"}` 일 때 자동 활성. 또는 `bx_style_on=True` 파라미터를 명시하면 임의 언어에도 강제 적용 가능.

#### 1.2.1 페르소나 (Persona)
> **📌 매핑 변수**: `BX_STYLE_RULES["system_identity"]`

- **자신감 있는 탐험가 (Confident Explorer)**: 기술 매뉴얼이 아닌 친근하고 자신감 있는 가이드처럼 영문 카피를 작성한다. OPEN/BOLD/AUTHENTIC 보이스를 아래 구체 기법으로 구현한다.

#### 1.2.2 핵심 보이스 속성 (Voice Attributes)
> **📌 매핑 변수**: `BX_STYLE_RULES["voice_attributes"]`
> **📌 참고**: 직역 금지·hedging 금지·격식체 지양 등 COMMON과 중복되는 원칙은 여기서 제외하고 COMMON에 일원화됨.

- **OPEN** (reveals, invites):
  1. Go beyond the literal benefit — reveal the experience, emotion, or new perspective behind the feature.
  2. Personify our tech to create intentional wit linked to product functionality. (e.g., 'This AI helps pay the bills')
  3. Upend expectations: set up a sentence one way, then give it an unexpected ending. (e.g., 'Visuals so real, real life looks fake.')

- **BOLD** (leads, takes a stance):
  1. Pair technical detail with the real emotional reaction it inspires. (e.g., 'Reaction: woah.')
  2. Play up contrast to create dramatic effect that underscores the different angles of our innovation. (e.g., 'Super small. Supremely smart.')
  3. Share our POV: take a clear stance rather than sitting on the fence.

- **AUTHENTIC** (grounds, connects):
  1. Write to a friend: imagine writing to someone you know. Replace technical language with everyday language.
  2. Find the upside: reframe negatives to positives to make our tech feel approachable. (e.g., 'Look forward to laundry day.')
  3. Find a tangible benefit: pull out a specific, relatable benefit instead of a broad claim. (e.g., 'Never run out of eggs again.')

#### 1.2.3 부정 제약 사항 (Negative Constraints)
> **📌 매핑 변수**: `BX_STYLE_RULES["negative_constraints"]`
> **📌 참고**: COMMON 중복 항목(직역 금지, hedging 금지, 격식 지양) 제거 후 BX 고유 제약만 유지.

- Do NOT use negative framing — always reframe into a positive benefit. (e.g., 'Don't worry about bills' → 'Enjoy savings')
- Never try too hard to relate — we're still a premium brand. Avoid slang or overly casual phrasing.
- Never be overly metaphorical — write with purpose and refinement.

---

## 2. 검수 관련 (Audit Criteria)

AI 검수 시 다음 6가지 항목을 기준으로 점수를 매깁니다.
> **📌 매핑 변수**: 
> - 검수 도입부: `AUDIT_INTRO`
> - 체크리스트: `AUDIT_CHECKLIST_RULES`
> - 채점 등급: `AUDIT_GRADE_CRITERIA` (`Excellent`, `Good`, `Needs Revision`)

1. **문법/유창성**: 오타, 문법 오류, 성수 일치, 관용구 사용 등 정밀 점검.
2. **원문의미 충실도**: 원문의 핵심 의미·뉘앙스·사용자 혜택이 번역에서 손실 없이 전달되었는지 확인. 직역 여부와 무관하게 '정보 손실' 또는 '의미 왜곡'이 발생했는지만 판단한다.
3. **용어집 준수**: 제공된 glossary 데이터와 100% 일치하는지 확인 (대소문자, 띄어쓰기 포함). 항목별 예외 규칙(rule/remark)이 있는 경우 예외가 우선 적용되었는지 확인. 현지화 자연스러움 여부와 관계없이 절대 적용되는 기준이다.
4. **현지화**: 해당 언어권 현지인이 실제로 사용하는 자연스러운 표현인지 종합 평가. ① [언어별 현지화 기준] 규칙 준수 (예: 독일어 Du-form, 일본어 ます형, 프랑스어 Tu/Vous 등), ② 직역·구조적 번역이 아닌 시장 맥락에 맞는 표현 선택, ③ 문화적 뉘앙스와 브랜드 보이스(Confident Explorer)의 현지 적용.
5. **대소문자 표기**: 대상 언어의 문장형(sentence case) 또는 타이틀형(title case) 등 일반 대소문자 표기 규칙 준수 여부.
6. **서식 및 표기**: [서식 규칙] 섹션 기준으로 점검: glossary 용어의 bracket 표기 적용 여부, 탐색 경로(nav path)의 따옴표 및 마침표 위치, 타이포그래피·구두점·간격 등 대상 언어 표기 규칙 준수 여부.

---

## 3. 언어별 상세 특화 규칙 (Detailed Localization Rules)

시스템은 아래와 같이 각 언어 및 시장별로 세분화된 규칙을 적용합니다.
> **📌 매핑 변수**: `LANGUAGE_LOCALIZATION_RULES` (`src/translation_web_app/prompt_modules.py`)

### 언어 매칭 전략
> **📌 생성 로직**: `PromptBuilder.get_language_rule()` (`src/translation_web_app/prompt_builder.py`)

`target_lang` 입력값을 아래 2단계로 매칭합니다.

1. **Exact match** (대소문자 무시): `"english_us"` → `English_US` ✓
2. **Fuzzy substring match** (최장 키 우선): 정확히 일치하는 키가 없으면, `LANGUAGE_LOCALIZATION_RULES`의 키를 길이 내림차순으로 정렬 후 `target_lang`에 포함된 키를 반환. 예: `"English_US_variant"` → `English_US` (`English`보다 먼저 매칭)
3. **매칭 실패**: 두 단계 모두 해당 없으면 언어 섹션 자체가 프롬프트에서 생략됩니다.

---

### 🇰🇷 KR (한국) - Korean (`Korean`)
> **💡 정책 요약**: 존댓말(honorific) 또는 문어체 스타일 일관성 유지. UI 전체 용어 일관성 확보.

```text
- Use consistent polite (honorific) or formal literary style as appropriate for the context.
- Ensure terminology consistency throughout the UI.
```

---

### 🌐 (글로벌 폴백) - English (`English`)
> **💡 정책 요약**: 특정 시장 변형(US/UK/AU/SG)이 명시되지 않고 단순히 "English"로만 입력될 경우 적용되는 폴백 규칙. US 영어 철자를 기본으로 하며 디스클레이머 마침표 위치도 US 기준을 따름.

```text
- Use US English spelling and wording (e.g., 'color', 'personalize') unless a more specific English market variant is specified.
- Follow the project-specific rule for disclaimers: place the sentence-ending period outside the closing quotation mark.
```

---

### 🇺🇸 US (미국) - English (`English_US`)
> **💡 정책 요약**: US 철자 사용 (color, personalize). 디스클레이머 마침표는 따옴표 밖(`".`)에 위치.

```text
- Use US English spelling and wording (e.g., 'color', 'personalize').
- Follow the project-specific rule for disclaimers: place the sentence-ending period outside the closing quotation mark.
```

---

### 🇬🇧 UK (영국) - English (`English_UK`)
> **💡 정책 요약**: British 철자 사용 (colour, personalise, optimise). US 전용 어휘 지양.

```text
- Use British English spelling (e.g., 'colour', 'personalise', 'optimise').
- Avoid US-specific vocabulary and phrasing.
```

---

### 🇦🇺 AU (호주) - English (`English_AU`)
> **💡 정책 요약**: British 철자 기반. 지나친 마케팅 톤을 지양하고 명확하고 유익한 톤 유지.

```text
- Use Australian English with British-style spelling.
- Avoid overly aggressive US-style marketing tones; keep it helpful and clear.
```

---

### 🇸🇬 SG (싱가포르) - English (`English_SG`)
> **💡 정책 요약**: 국제 싱가포르 영어(British 철자) 사용. 현지 속어(Singlish) 사용 금지.

```text
- Use concise, international Singapore English with British-style spelling where appropriate.
- Do NOT use Singlish, local slang, or overly US-specific wording.
```

---

### 🇫🇷 FR (프랑스) - French (`French`)
> **💡 정책 요약**: `Vous-form` 일관 사용. 따옴표는 `«...»` (길르메) 사용. 불필요한 대문자 사용 지양.

```text
- Use Vous-form consistently; do not use Tu-form.
- Use «...» (guillemets) for quoted text and navigation paths.
- Avoid unnecessary capitalization in UI copy.
- Use natural French phrasing and avoid English-influenced structures.
```

---

### 🇧🇪 BE (벨기에) - French (`French_BE` / `French_Belgium`)
> **💡 정책 요약**: `Vous-form` 일관 사용. 따옴표는 `«...»` (길르메) 사용. 벨기에 시장에 적합한 중립적 톤 유지. 프랑스 본토 전용 관용구 지양.

```text
- Use Vous-form consistently; do not use Tu-form.
- Use «...» (guillemets) for quoted text and navigation paths.
- Use neutral French and avoid overly idiomatic expressions specific to mainland France.
- Ensure consistent tone for the Belgian market.
```

---

### 🇨🇦 CA (캐나다) - French (`French_CA` / `French_Canada`)
> **💡 정책 요약**: `Vous-form` 일관 사용. 따옴표는 `«...»` (길르메) 사용. 북미 프랑스어 표준 및 자연스러운 어구 우선.

```text
- Use Vous-form consistently; do not use Tu-form.
- Use «...» (guillemets) for quoted text and navigation paths.
- Follow Canadian French standards; prioritize phrasing natural to North American French over mainland France idioms.
```

---

### 🇩🇪 DE (독일) - German (`German`)
> **💡 정책 요약**: `Du-form` 기본 사용. 명사 대문자 표기 및 복합어 구조 준수. 기술적 어투 지양. 따옴표는 `„..."` (독일식 길르메) 사용.

```text
- Use Du-form consistently unless the locale or project explicitly requires Sie-form.
- Ensure natural capitalization of nouns and maintain natural compound word structures.
- Avoid overly formal or technical wording in short UI copy.
- Use „..." (German quotation marks) for quoted text and navigation paths; do not use English straight quotation marks.
```

---

### 🇮🇹 IT (이탈리아) - Italian (`Italian`)
> **💡 정책 요약**: 자연스러운 이탈리아어 문장 구조 사용. 영어식 명사 나열 지양.

```text
- Use natural Italian UI sentence structures; avoid English-style noun-chaining.
```

---

### 🇪🇸 ES (스페인) - Spanish (`Spanish` / `Spanish_ES`)
> **💡 정책 요약**: 카스티야(Spain) 스페인어 및 `Usted` 기본 사용. 라틴 아메리카 전용 어휘 지양.

```text
- Use Usted consistently unless the locale or project explicitly requires Tú.
- Use Spain Spanish (Castilian) and avoid Latin American-specific wording or usage.
- Keep Spanish regionally neutral unless a market-specific variant is requested.
```

---

### 🇳🇱 NL (네덜란드) - Dutch (`Dutch`)
> **💡 정책 요약**: `u/uw` 공식체 일관 사용. 직접적이고 간결한 문체. 영어식 어순이나 명사구 구조 지양.

```text
- Use 'u/uw' (formal address) consistently; do not use je/jij.
- Keep Dutch copy direct and concise.
- Avoid literal translation of English word order or noun-phrase structures.
```

---

### 🇸🇪 SE (스웨덴) - Swedish (`Swedish`)
> **💡 정책 요약**: 간결한 UI copy. 영어식 Title Case 대신 Sentence Case(첫 글자만 대문자) 우선.

```text
- Keep Swedish UI copy concise and natural.
- Avoid English-style title case; prioritize sentence case for headings and buttons.
```

---

### 🇦🇪 AE (아랍에미리트) - Arabic (`Arabic`)
> **💡 정책 요약**: 현대 표준 아랍어(MSA) 사용. UI 방향성(RTL), 구두점, 문장 끝맺음 규칙 준수.

```text
- Use Modern Standard Arabic (MSA).
- Follow Arabic conventions for UI directionality, punctuation, and sentence-ending styles.
- Avoid literal translations that sound unnatural in Arabic.
```

---

### 🇵🇹 PT (포르투갈) - European Portuguese (`European Portuguese`)
> **💡 정책 요약**: 유럽 포르투갈어 전용 어휘 사용. 브라질식 어휘나 진행형(gerund) 구조 지양.

```text
- Use European Portuguese consistently; avoid Brazilian Portuguese wording, vocabulary, or gerund structures.
```

---

### 🇧🇷 BR (브라질) - Brazilian Portuguese (`Brazilian Portuguese`)
> **💡 정책 요약**: 브라질 포르투갈어 전용 어휘 및 문구 사용. 유럽식 어휘/구조 지양.

```text
- Use Brazilian Portuguese consistently; avoid European Portuguese vocabulary or phrasing.
```

---

### 🇷🇺 RU (러시아) - Russian (`Russian`)
> **💡 정책 요약**: 러시아어 어순 준수. 영어식 명사구 직역 및 과도한 대문자 지양. 따옴표는 `«...»` (길르메) 사용.

```text
- Use natural Russian word order.
- Avoid English-style noun-phrase literal translations and excessive capitalization.
- Use «...» (guillemets) for quoted text and navigation paths; do not use English straight quotation marks.
```

---

### 🇹🇷 TR (터키) - Turkish (`Turkish`)
> **💡 정책 요약**: 터키어 어순 준수. UI에 적합한 간결한 명령형 또는 묘사형 사용.

```text
- Use natural Turkish word order and avoid structural calques from English.
- Maintain concise imperative or descriptive forms suitable for UI copy.
```

---

### 🇨🇳 CN (중국) - Simplified Chinese (`Simplified Chinese`)
> **💡 정책 요약**: 중국 본토 표준 용어 및 간체자 사용. 대만식 용어 지양. 전각 구두점 사용. 따옴표는 전각 `"..."` 사용 (`「」` 금지).

```text
- Use natural Mainland Chinese wording and Simplified Chinese characters.
- Avoid Taiwan-specific terminology.
- Use fullwidth punctuation marks (，。！？) consistently; avoid half-width ASCII punctuation.
- Use fullwidth double quotation marks ("...") for quoted text and navigation paths; do not use 「」 (Japanese-style brackets).
```

---

### 🇹🇼 TW (대만) - Traditional Chinese (`Traditional Chinese`)
> **💡 정책 요약**: 대만 표준 용어 및 번체자 사용. 중국 본토식 용어 지양. 전각 구두점 사용. 따옴표는 `「...」` 사용.

```text
- Use natural Taiwan Traditional Chinese wording and characters.
- Avoid Mainland Chinese-specific terminology.
- Use fullwidth punctuation marks (，。！？) consistently; avoid half-width ASCII punctuation.
- Use 「...」 for quoted text and navigation paths.
```

---

### 🇯🇵 JA (일본) - Japanese (`Japanese`)
> **💡 정책 요약**: `ます-form` 기본 사용. 자연스러운 UI 표현(操作/設定 등) 우선. 직역형 구조 지양. 내비게이션 경로는 `「 」` 사용.

```text
- Use consistent ます-form unless project guidance specifies otherwise.
- Use natural Japanese UI phrasing and avoid close structural calques from English or Korean.
- Prioritize natural '操作/設定' style expressions over mechanical literal translations.
- Avoid excessive honorifics unless the context clearly requires them.
```

---

### 🇵🇱 PL (폴란드) - Polish (`Polish`)
> **💡 정책 요약**: 격 변화(declension) 및 문법적 일치 준수. 영어식 명사 나열 지양.

```text
- Ensure natural Polish declension and grammatical agreement.
- Avoid English-style noun-chaining; use natural phrasing.
```

---

### 🇻🇳 VN (베트남) - Vietnamese (`Vietnamese`)
> **💡 정책 요약**: 자연스러운 어순 및 간결한 UI 표현. 영어식 대문자 패턴 지양.

```text
- Use natural Vietnamese word order and concise UI phrasing.
- Avoid English-style capitalization patterns.
```

---

### 🇹🇭 TH (태국) - Thai (`Thai`)
> **💡 정책 요약**: 자연스러운 태국어 UI 표현. 불필요한 공백 및 구두점 복사 금지.

```text
- Use natural Thai UI phrasing.
- Avoid unnecessary spaces and mechanical copying of English punctuation.
```

---

### 🇮🇩 ID (인도네시아) - Indonesian (`Indonesian`)
> **💡 정책 요약**: 간결하고 자연스러운 문체. 과하게 격식적인 구조나 영어식 직역 지양.

```text
- Keep Indonesian copy concise and natural.
- Avoid overly formal structures or English-style literal phrasing.
```

---

## 4. 용어집 관련 (Glossary Rules)

**Purpose**  
제품명, 기능명, 메뉴명, 브랜드 표현이 화면마다 다르게 번역되지 않도록 용어 기준을 고정합니다. SmartThings처럼 동일한 기능과 메뉴가 여러 화면에서 반복되는 서비스에서는 용어가 조금만 달라져도 사용자가 다른 기능으로 오해할 수 있기 때문에, glossary를 기준으로 명칭 일관성을 유지하는 것이 핵심 목적입니다.

**Feature**  
Glossary에 등록된 용어는 번역 참고 자료가 아니라 필수 준수 기준으로 적용됩니다. AI가 문장을 자연스럽게 다듬더라도 등록 용어의 대소문자, 띄어쓰기, 시장별 표기는 그대로 유지하며, 문맥에 따라 괄호 적용 여부만 다르게 제어합니다.

> **📌 매핑 변수**: 
> - 기본 용어집 규칙: `GLOSSARY_TERM_RULES`
> - 브래킷 래핑 규칙: `GLOSSARY_BRACKET_WRAP_RULE`
> - 대괄호 수동 제외 키워드: `GLOSSARY_EXEMPT_MARKERS` (`["no bracket", "대괄호 제외", "괄호 제외"]`)
> - 타이틀/버튼 대괄호 자동 제외: `GLOSSARY_NO_BRACKET_INSTRUCTION`

> **📌 참고**: §4(용어집)와 §5(타이포)는 문서상 별도 섹션이지만, 실제 프롬프트 출력 시 `_build_formatting_section()`이 두 섹션을 `[GLOSSARY RULES]` → `[Typography and Punctuation Rules]` 순서로 하나의 블록으로 조립한다.

### 4.1 기본 용어집 사용 규칙

**Purpose**  
AI가 "더 자연스러워 보인다"는 이유로 핵심 용어를 임의로 바꾸지 않게 합니다. 예를 들어 `SmartThings Find`처럼 제품 정책상 고정된 표현은 제목, 버튼, 설명문 어디에 나오더라도 등록된 표기를 우선합니다.

**Feature**  
지정 용어를 문자 단위로 보존하고, 시장별 variant와 개별 예외 규칙을 함께 적용합니다. 즉, 용어집에 등록된 단어는 의미만 맞추는 대상이 아니라 검수 시 실제 문자열이 일치해야 하는 관리 대상입니다.

> **📌 매핑 변수**: `GLOSSARY_TERM_RULES`

```text
- Use provided glossary terms exactly as given — including capitalization, spacing, and market variants. Glossary capitalization is authoritative and overrides title case, sentence case, and heading/button capitalization rules; do not adapt glossary terms for naturalness.
- Apply term-specific rule or remark exceptions before generic formatting rules.
```

> **📌 참고**: 두 번째 줄은 `src/translation_web_app/prompt_builder.py` `_build_formatting_section()`에 하드코딩된 문자열로, 개별 용어의 `rule` 필드(예: "대괄호 제외", "비활성화") 마커가 일반 포맷팅 규칙보다 먼저 처리되어야 함을 LLM에 지시한다.

---

### 4.2 문맥(row_key) 기반 브래킷 분기

**Purpose**  
같은 용어라도 UI 위치에 따라 보이는 방식이 달라져야 합니다. 제목이나 버튼은 짧고 간결해야 하므로 괄호를 제거하고, 설명문이나 고지 문구에서는 핵심 용어를 더 명확히 식별할 수 있도록 괄호를 적용합니다.

**Feature**  
`row_key`를 기준으로 UI 문맥을 자동 판별하여 `title_button`, `description`, `disclaimer` 모드로 나눕니다. 이를 통해 버튼/헤딩은 깔끔하게 유지하고, 본문/고지 영역은 용어 강조와 검수 편의성을 확보합니다.

> **📌 생성 로직**: `PromptBuilder.get_glossary_context_mode()` (`src/translation_web_app/prompt_builder.py`)

| 판별 순서 | 조건 | 결과 모드 |
|---------|------|---------|
| 1 | `"disclaimer"` 포함 | `disclaimer` |
| 2 | `"description"` 포함 | `description` |
| 3 | `"title"` 또는 `"button"` 포함, 또는 **숫자로 끝남** (`\d+$`) | `title_button` |
| 4 | 해당 없음 | `description` (기본값) |

> **📌 숫자 종료 규칙**: `row_key`가 숫자로 끝나는 경우 (예: `field_1`, `label_42`) 자동으로 `title_button` 모드로 처리되어 브래킷 없이 출력됩니다.

#### `title_button` 모드

**Purpose**  
제목, 섹션 헤딩, 버튼 문구는 사용자가 빠르게 훑고 바로 이해해야 하는 영역입니다. 용어를 괄호로 감싸면 UI가 무거워 보이거나 클릭 요소처럼 오해될 수 있어, 간결한 화면 표현을 우선합니다.

**Feature**  
Glossary 용어의 텍스트 자체는 그대로 유지하되 `[]`, `「」` 등 주변 기호는 제거합니다. 대소문자도 제목/버튼 스타일에 맞춰 바꾸지 않고 glossary 표기를 그대로 따릅니다.

> **📌 매핑 변수**: `GLOSSARY_NO_BRACKET_INSTRUCTION`

```text
- For title, section heading, and button copy, do not wrap glossary terms in any brackets. Use the glossary term text exactly as provided, without [], 「」, or any other surrounding bracket marks, even if the source text contains brackets. Do not change glossary capitalization to satisfy heading or button case style.
```

#### `description` 모드 (기본값)

**Purpose**  
설명문은 기능 안내, 설정 설명, 사용 조건처럼 정보량이 많은 영역입니다. 이 안에서 핵심 제품·기능 용어가 묻히지 않도록 시각적으로 구분하는 것이 목적입니다.

**Feature**  
Glossary 용어를 대상 언어에 맞는 괄호로 감쌉니다. 일본어는 `「 」`, 그 외 언어는 `[ ]`를 기본으로 사용해 용어 적용 여부를 쉽게 확인할 수 있게 합니다.

> **📌 매핑 변수**: `GLOSSARY_BRACKET_WRAP_RULE`

브래킷 종류는 `get_brackets(target_lang)`으로 결정됩니다.

| 언어 | 브래킷 | 판별 조건 |
|------|--------|---------|
| Japanese | `「 」` | `target_lang`에 `"Japanese"` 또는 `"일본"` 포함 |
| 그 외 전체 | `[ ]` | 위 조건 해당 없음 |

**프롬프트 출력 (한국어·영어·서양권 등):**

```text
- Use provided glossary terms exactly as given — including capitalization, spacing, and market variants. Glossary capitalization is authoritative and overrides title case, sentence case, and heading/button capitalization rules; do not adapt glossary terms for naturalness.
- Apply term-specific rule or remark exceptions before generic formatting rules.
- Wrap glossary terms in '[' and ']'.
```

**프롬프트 출력 (일본어 JA):**

```text
- Use provided glossary terms exactly as given — including capitalization, spacing, and market variants. Glossary capitalization is authoritative and overrides title case, sentence case, and heading/button capitalization rules; do not adapt glossary terms for naturalness.
- Apply term-specific rule or remark exceptions before generic formatting rules.
- Wrap glossary terms in '「' and '」'.
```

#### `disclaimer` 모드

**Purpose**  
Disclaimer는 조건, 제한, 안내 사항을 정확히 전달해야 하는 영역입니다. 일반 용어는 명확히 드러내되, 사용자가 실제 앱에서 따라가야 하는 메뉴 경로는 화면 표기와 다르게 보이지 않도록 보호합니다.

**Feature**  
`row_key`를 소문자로 정규화했을 때 `"disclaimer"` 문자열이 포함되면 최우선으로 `disclaimer` 모드가 선택됩니다. 본문 안의 glossary 용어는 description 모드처럼 괄호로 감싸지만, `Settings > Device` 같은 nav path 내부 용어는 괄호를 적용하지 않습니다. 이를 통해 용어 강조와 메뉴 경로 정확성을 동시에 유지합니다.

> **📌 매핑 변수**: `GLOSSARY_BRACKET_WRAP_RULE` + `GLOSSARY_DISCLAIMER_NAV_EXCEPTION`

> **📌 판별 조건**: `row_key.lower()`에 `"disclaimer"`가 포함되는 경우. 예: `disclaimer`, `device_disclaimer`, `settings_disclaimer_01`

브래킷은 `description` 모드와 동일하게 적용하되, nav path 예외 규칙이 추가됩니다.

```text
- Wrap glossary terms in '[' and ']'. Exception: do not wrap terms inside navigation paths (e.g., Settings > Device).
```

---

### 4.3 용어별 브래킷 수동 제외 (Per-Term Exemption)

**Purpose**  
자동 규칙만으로 처리하기 어려운 예외 용어를 운영자가 직접 관리할 수 있게 합니다. 브랜드명, 앱명, 이미 고유명사처럼 굳어진 표현은 설명문 안에서도 괄호 없이 보여주는 편이 더 자연스러울 수 있습니다.

**Feature**  
엑셀 용어집의 `rule` 또는 `remark` 열에 제외 마커를 입력하면 해당 용어는 문맥과 무관하게 브래킷 적용 대상에서 빠집니다. 코드 수정 없이 용어집 데이터만으로 예외를 관리할 수 있습니다.

> **📌 매핑 변수**: `GLOSSARY_EXEMPT_MARKERS`

```python
GLOSSARY_EXEMPT_MARKERS = ["no bracket", "대괄호 제외", "괄호 제외"]
```

엑셀 용어집의 rule/remark 열에 위 문자열 중 하나를 입력하면 해당 용어의 브래킷이 자동 면제됩니다.

---

### 4.4 용어집 없음 (No Glossary Available)

**Purpose**  
용어집이 제공되지 않은 작업에서 AI가 임의로 glossary를 추측하거나 없는 기준을 만들어 적용하지 않게 합니다.

**Feature**  
`glossary_context`가 없으면 용어집 관련 지시를 아래 fallback 문구로 대체합니다. 이 경우에는 일반 현지화 규칙과 언어별 규칙만 적용하며, 용어집 기반 강제 표기는 수행하지 않습니다.

```text
No glossary terms are provided for this source text.
```

---

## 5. 타이포그래피 및 서식 규칙 (Typography & Formatting)

**Purpose**  
번역된 문장이 의미만 맞는 수준을 넘어, 실제 현지 UI처럼 자연스럽게 보이도록 표기 품질을 관리합니다. 따옴표, 마침표 위치, 공백, 전각/반각 기호, RTL 방향성이 어긋나면 사용자는 번역이 기계적이거나 낯설다고 느낄 수 있습니다.

**Feature**  
메뉴 경로(nav path), 고지 문구(disclaimer), 따옴표와 마침표 위치처럼 UI에서 반복적으로 노출되는 표기를 언어별 관행에 맞춰 제어합니다. 특히 사용자가 실제 화면에서 따라가야 하는 메뉴 경로는 각 언어권에 익숙한 기호로 감싸 구분성을 높입니다.

> **📌 매핑 변수**: 
> - 타이포그래피 기본 규칙: `TYPOGRAPHY_AND_PUNCTUATION_RULES`
> - Nav path 예외 처리: `GLOSSARY_DISCLAIMER_NAV_EXCEPTION`
> - Nav path 범용 규칙: `GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE`
> - Nav path JA 전용 규칙: `GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_JA`

- **Typography 기본 규칙**: 모든 번역/검수 프롬프트에 `[Typography and Punctuation Rules]` 블록으로 삽입됨.
  > **📌 매핑 변수**: `TYPOGRAPHY_AND_PUNCTUATION_RULES`
  > **Purpose**: 영어 원문의 구두점, 따옴표, 띄어쓰기 방식을 그대로 복사하지 않고, 타겟 언어권에서 자연스러운 표기 관행을 따르게 합니다.
  > **Feature**: 대상 언어와 locale에 맞는 punctuation, spacing, quotation mark convention을 적용하도록 기본 지시를 제공합니다. 이 규칙은 glossary 유무와 관계없이 `_build_formatting_section()` 마지막에 항상 추가됩니다.

  ```text
  [Typography and Punctuation Rules]
  - For quotation marks, spacing, and punctuation specific to the target language, follow the rules stated in the language section above; where no specific rule is given, apply the standard convention for that language and locale.
  - Do not mechanically copy English punctuation, quotation mark placement, spacing, or sentence-ending style into other languages.
  ```

  > **📌 참고**: 첫 번째 규칙은 §3 언어별 룰(예: French `«»`, German `„..."`, 중문 전각)을 Typography 섹션에서 명시적으로 위임한다. 언어 룰이 없는 경우 해당 언어 표준 관행이 폴백으로 적용된다.

- **Nav Path 괄호 예외** (`GLOSSARY_DISCLAIMER_NAV_EXCEPTION`): disclaimer 모드에서 bracket wrap rule 뒤에 append되어 nav path 내부 용어에 브래킷을 적용하지 않도록 지시합니다.

  ```text
  Exception: do not wrap terms inside navigation paths (e.g., Settings > Device).
  ```

  > **📌 참고**: 단독 삽입이 아니라 `GLOSSARY_BRACKET_WRAP_RULE` 뒤에 이어 붙여 한 줄로 출력된다. (예: `Wrap glossary terms in '[' and ']'. Exception: do not wrap terms inside navigation paths (e.g., Settings > Device).`)

- **Nav Path 따옴표 및 마침표 위치 (범용 — 일본어 제외)** (`GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE`): disclaimer 행에서 항상 삽입. 따옴표 종류는 대상 언어별 룰(§3)에 위임되며, 마침표는 닫는 따옴표 밖에 배치.
  - 출력 예시: `"Settings > General".` / `«Einstellungen > Allgemein».`

  ```text
  Enclose navigation paths in quotation marks appropriate for the target language. Place the sentence-ending period outside the closing quotation mark.
  ```

- **Nav Path 따옴표 (East Asian — 마침표 지시 없음)** (`GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_EAST_ASIAN`): Japanese / Simplified Chinese / Traditional Chinese disclaimer 행에 적용. 마침표 위치 지시를 제거한 이유: 동아시아어는 문장이 path만으로 끝나지 않거나(JA: `「path」で設定できます。`, TW: `「path」中設定。`) 라벨 포맷으로 마침표 자체가 없으므로(CN: `开启路径："path"`) period placement 규칙이 의미 없음.
  - 판별 조건 — `target_lang`에 `"Japanese"`, `"일본"`, `"Chinese"`, `"중국"`, `"Taiwan"`, `"대만"` 중 하나 포함

  ```text
  Enclose navigation paths in quotation marks appropriate for the target language.
  ```

**disclaimer 모드 최종 프롬프트 출력 (East Asian — JA 예시):**

```text
- Wrap glossary terms in '「' and '」'. Exception: do not wrap terms inside navigation paths (e.g., Settings > Device).
- Enclose navigation paths in quotation marks appropriate for the target language.
```

**disclaimer 모드 최종 프롬프트 출력 (비 East Asian):**

```text
- Wrap glossary terms in '[' and ']'. Exception: do not wrap terms inside navigation paths (e.g., Settings > Device).
- Enclose navigation paths in quotation marks appropriate for the target language. Place the sentence-ending period outside the closing quotation mark.
```

---

## 6. 번역 및 검수 최종 프롬프트 아키텍처 및 해설 (Prompt Architecture & Raw Templates)

**Purpose**  
앞에서 정의한 번역, 현지화, 용어집, 서식 규칙이 실제 AI 프롬프트 안에서 어떤 순서로 사용되는지 보여줍니다. 1~5번이 "무엇을 지켜야 하는가"를 설명한다면, 6번은 그 기준을 AI에게 어떻게 전달하고 결과를 어떻게 받는지 설명합니다.

**Feature**  
프롬프트는 하나의 긴 지시문이 아니라 Persona, Common, Language, BX, RAG, Formatting, Output처럼 역할별 모듈로 조립됩니다. 이 구조 덕분에 대상 언어, 문맥, 용어집, RAG 유무에 따라 필요한 규칙만 선택적으로 삽입할 수 있습니다.

시스템은 `src/translation_web_app/prompt_builder.py`의 `PromptBuilder` 클래스를 통해 앞서 정의된 상수를 조합하여 동적으로 프롬프트를 생성합니다. 프롬프트 코드 원문(Raw Text)을 복사할 때 어떠한 마크다운/HTML 태그도 딸려가지 않도록 원문은 순수 파이썬 코드 블록으로 유지합니다.

### 6.1 번역 프롬프트 구조 및 원문 (Translation Prompt)
> **📌 생성 메서드**: `PromptBuilder.build_translation_prompt()`

**Purpose**  
AI가 번역해야 할 대상 언어, 번역 방향, 품질 기준, 출력 형식을 한 번에 이해하도록 구성합니다. 원문의 의미와 뉘앙스를 유지하면서도 현지 원어민이 자연스럽게 읽는 결과물을 만드는 것이 목표입니다.

**Feature**  
번역 프롬프트는 대상 언어, BX 스타일 적용 여부, RAG 유사도 매칭 여부, 문맥(`row_key`)에 따라 7개 모듈로 조립됩니다. 앞쪽 모듈은 역할과 톤을 설정하고, 뒤쪽 모듈은 용어집·서식·JSON 출력 형식을 고정합니다.

#### [1] Persona Section (역할 및 기본 태스크 정의)
**Purpose**  
대상 언어(Target Language)와 출발어(Source Language)를 설정하고, 번역가로서의 기본 정체성을 부여합니다. 원문의 의미와 뉘앙스를 온전히 전달하면서, 현지 원어민이 자연스럽게 읽을 수 있는 결과물을 생성하는 것을 목표로 합니다.

**Feature**  
Role(누구) + Languages(무엇에서 무엇으로) + Task(어떻게) 3요소로 구성되어 AI가 과제의 범위를 즉시 파악할 수 있습니다. BX 모드에서는 역할이 `Samsung BX Writer & Translator`로 확장되어 번역뿐 아니라 브랜드 톤에 맞춘 문장 polish까지 수행합니다.

```python
# [1] Persona Section (역할 부여)
# 일반 모드 시:
You are a professional {target_lang} localizer.
Source Language: {source_lang}
Target Language: {target_lang}
TASK: Translate the source text naturally for a native speaker while faithfully conveying the full intent and nuance of the original.

# BX 모드 시:
You are the Samsung BX Writer & Translator.
Source Language: {source_lang}
Target Language: {target_lang}
TASK: Translate and polish the source text naturally for a native speaker while faithfully conveying the full intent and nuance of the original.
```

#### [2] Common Section (공통 현지화 표준)
**Purpose**  
언어와 시장이 달라져도 공통으로 지켜야 하는 현지화 품질 기준을 제공합니다. 단순히 단어를 바꾸는 번역이 아니라, 원문의 의도와 사용자 혜택이 현지 언어에서도 자연스럽게 전달되도록 합니다.

**Feature**  
의미 보존, 직역 지양, 친근한 기술 표현, 문화적으로 어색한 표현 회피, UI copy 간결성이라는 기본 원칙을 한 번에 주입합니다.

```python
# [2] Common Section (공통 품질 기준)
[COMMON LOCALIZATION STANDARD]
- Preserve the original intent, nuance, and user benefit of the source.
- Avoid overly literal translation when more natural, market-appropriate wording communicates the same intent better.
- Where appropriate, use creative expression — refined wit, metaphor, or personification — to make technology feel approachable rather than technical.
- Avoid culturally awkward idioms, fear-based framing, overly formal or technical language, and hedging words (e.g., 'hopefully', 'try to', 'might').
- Keep UI copy concise while keeping the action or benefit clear.
```

#### [3] Language Section (언어별 상세 특화 규칙)
**Purpose**  
각 언어권 사용자가 실제로 익숙하게 받아들이는 말투와 표기 방식을 반영합니다. 같은 의미라도 언어마다 존댓말, 어순, 따옴표, 대소문자 관행이 다르기 때문에 언어별 기준을 별도로 제공합니다.

**Feature**  
대상 언어가 시스템에 정의된 언어 규칙과 매칭되면 해당 규칙이 자동 삽입됩니다. 예를 들어 일본어는 `ます`형, 독일어는 `Du-form`, 프랑스어는 `Vous-form` 같은 언어별 UI 관행을 반영합니다.

```python
# [3] Language Section (언어별 특화 규칙 - 해당 언어 매칭 시 삽입)
[LANGUAGE SPECIFIC RULE]
{Language_Label} (예: Japanese ます-form Consistency)
- {Language_Specific_Rule_1}
- {Language_Specific_Rule_2}
```

#### [4] BX Style Section (삼성 BX 브랜드 보이스 주입)
**Purpose**  
영문 결과물이 단순히 정확한 번역을 넘어 삼성 브랜드 보이스처럼 들리도록 합니다. 기술 설명이 딱딱한 매뉴얼처럼 보이지 않고, 자신감 있고 친근한 가이드처럼 느껴지게 만드는 것이 목적입니다.

**Feature**  
`target_lang`이 `English_US` 또는 `English`일 때 자동 삽입됩니다. Confident Explorer 페르소나와 OPEN/BOLD/AUTHENTIC 보이스 속성, 부정 제약, Few-shot 예시를 함께 제공해 영문 카피의 톤을 조정합니다.

```python
# [4] BX Style Section (target_lang이 English_US/English일 때 자동 삽입)
[SAMSUNG BX STYLE]
Persona: Confident Explorer (자신감 있는 탐험가)
Goal: Craft English copy that sounds like a confident, friendly guide — not a tech manual. Apply OPEN, BOLD, AUTHENTIC voice through specific techniques below.
Target Language: {target_lang}

Voice Attributes:
- OPEN:
  - Go beyond the literal benefit to reveal all the hidden dimensions — the experience, emotion, or new perspective behind the feature.
  - Personify our tech to create intentional wit linked to product functionality. (e.g., 'This AI helps pay the bills')
  - Upend expectations: set up a sentence one way, then give it an unexpected ending. (e.g., 'Visuals so real, real life looks fake.')
- BOLD:
  - Pair technical detail with the real emotional reaction it inspires. (e.g., 'Reaction: woah.')
  - Play up contrast to create dramatic effect that underscores the different angles of our innovation. (e.g., 'Super small. Supremely smart.')
  - Share our POV: take a clear stance rather than sitting on the fence.
- AUTHENTIC:
  - Write to a friend: imagine writing to someone you know. Replace technical language with everyday language.
  - Find the upside: reframe negatives to positives to make our tech feel approachable. (e.g., 'Look forward to laundry day.')
  - Find a tangible benefit: pull out a specific, relatable benefit instead of a broad claim. (e.g., 'Never run out of eggs again.')

Negative Constraints:
- Do NOT use negative framing — always reframe into a positive benefit. (e.g., 'Don't worry about bills' → 'Enjoy savings')
- Never try too hard to relate — we're still a premium brand. Avoid slang or overly casual phrasing.
- Never be overly metaphorical — write with purpose and refinement.

Few-shot Examples:
Type: OPEN (Headlines)
Input: Turn on the lights to create the perfect mood. (완벽한 분위기를 위해 조명을 켜세요)
Output: Lights? On. Mood? Up.
...
```

#### [5] RAG Context Section (번역 메모리 참조)
**Purpose**  
기존에 번역된 유사 문장을 참고해 화면 간 표현 차이를 줄입니다. 새로운 번역이 기존 앱 UI와 동떨어져 보이지 않도록 스타일과 용어 흐름을 맞추는 역할입니다.

**Feature**  
RAG 데이터베이스에서 유사 번역 메모리(TM)가 조회되면 `[Translation Memory Examples]` 섹션으로 삽입됩니다. AI는 이를 참고해 기존 표현 방식과 용어 일관성을 유지합니다.
> **📌 삽입 조건**: `rag_context`가 truthy(non-None, non-empty)일 때만 삽입. `[Translation Memory Examples]` 헤더가 없으면 자동 추가 (`_normalize_rag_section()`).

```python
# [5] RAG Context Section (RAG DB 유사 번역 메모리 존재 시 삽입)
[Translation Memory Examples]
{RAG_Matching_Examples}
Use these examples as style and terminology reference to maintain consistency.
```

#### [6] Formatting & Glossary Section (용어집 및 표기 규칙 제어)
**Purpose**  
번역 결과가 자연스럽더라도 용어집, 괄호, 메뉴 경로, 구두점 규칙이 흐트러지지 않도록 마지막 표기 기준을 적용합니다. 화면에 표시될 문자열의 일관성과 검수 가능성을 높이는 단계입니다.

**Feature**  
§4 용어집 규칙과 §5 타이포그래피 규칙이 하나의 블록으로 조립됩니다. `row_key`와 `target_lang`에 따라 glossary 용어의 괄호 적용 방식, nav path 예외, 언어별 구두점 규칙을 동적으로 결정합니다.

```python
# [6] Formatting & Glossary Section (용어집 및 서식 제어 기본 구조)
[GLOSSARY RULES]
- Use provided glossary terms exactly as given — including capitalization, spacing, and market variants. This overrides any localization style preference; do not adapt glossary terms for naturalness.
- Apply term-specific rule or remark exceptions before generic formatting rules.
```

> **📌 참고**: `GLOSSARY_TERM_RULES["rules"][1]` 상수(`Glossary capitalization is authoritative...`)는 `src/translation_web_app/prompt_modules.py`에 정의되어 있으나, 현재 조립 로직(`src/translation_web_app/prompt_builder.py` line 413)에서는 위 하드코딩 문구가 대신 삽입됩니다.

##### 🔹 문맥(`row_key`) 및 타겟 언어에 따른 동적 서식 분기 프롬프트

**1) Title & Button 문맥** (`title`, `button` 등 포함 시)
> UI 요소의 깔끔한 렌더링을 위해 괄호 마커를 일절 제외하고 원문 단어 그대로 출력하도록 강제합니다.
```text
- For title, section heading, and button copy, do not wrap glossary terms in any brackets. Use the glossary term text exactly as provided, without [], 「」, or any other surrounding bracket marks, even if the source text contains brackets. Do not change glossary capitalization to satisfy heading or button case style.
```

**2) Description / Body 문맥** (일반 설명문)
> 타겟 언어권의 관행에 맞는 기호로 용어집 단어를 강조(Wrapping)합니다.
```text
# 한국어, 영어, 서양권 언어 등:
- Wrap glossary terms in '[' and ']'.

# 일본어 (JA):
- Wrap glossary terms in '「' and '」'.
```

**3) Disclaimer 문맥** (`disclaimer` 포함 시)
> 법적 고지 문구 및 메뉴 탐색 경로(`Settings > General`) 안내 시, 괄호 래핑을 면제하고 언어에 맞는 따옴표 및 마침표 위치를 제어합니다.
```text
# 공통 예외 규칙 삽입:
- Exception: do not wrap terms inside navigation paths (e.g., Settings > Device).

# 범용 규칙 (JA 제외 전체 언어):
- Enclose navigation paths in quotation marks appropriate for the target language.
  Place the sentence-ending period outside the closing quotation mark.
  (출력 예시 EN: "Settings > General".)
  (출력 예시 DE: „Einstellungen > Allgemein".)
  (출력 예시 FR/RU: «Paramètres > Général».)

# JA (일본) 전용 규칙:
- Enclose navigation paths in 「 and 」.
  (출력 예시: 「設定 > 一般」から設定できます。)
```

##### 🔹 타이포그래피 및 구두점 규칙 (Typography and Punctuation Rules)
> 영어 원문(Source)의 구두점이나 띄어쓰기 관행을 기계적으로 복사하지 않고, 타겟 언어의 표준 맞춤법과 로케일(Locale) 관행을 준수하도록 지시합니다.
```text
[Typography and Punctuation Rules]
- Follow punctuation, spacing, and quotation mark conventions standard for the target language and locale.
- Do not mechanically copy English punctuation, quotation mark placement, spacing, or sentence-ending style into other languages.
```
> **📌 주요 언어별 동작 효과**:
> - **프랑스어 (`FR`)**: 콜론(`:`), 세미콜론(`;`), 물음표(`?`), 느낌표(`!`) 앞에 공백(Non-breaking space) 삽입 표준 준수.
> - **아랍어 (`AE`)**: 오른쪽에서 왼쪽(RTL) 방향성에 맞춰 아랍어 전용 쉼표(`،`) 및 물음표(`؟`) 사용.
> - **중국어/일본어 (`CN`, `TW`, `JA`)**: 서양식 반각 기호와 띄어쓰기 대신 전각 기호(`，`, `。`, `、`) 사용.

#### [7] Output Section (JSON 출력 규격 강제)
**Purpose**  
AI 응답에서 번역문만 안정적으로 추출할 수 있게 합니다. 사람이 읽는 설명형 답변이 아니라, 후처리 파이프라인이 바로 사용할 수 있는 결과물을 받는 것이 목적입니다.

**Feature**  
응답을 `{"translation": "..."}` 형태의 JSON 객체로 제한합니다. 불필요한 설명이나 사족이 섞이지 않아 시스템에서 번역 결과를 안정적으로 파싱할 수 있습니다.

```python
# [7] Output Section (출력 규격 강제)
OUTPUT: Return ONLY a JSON object with a "translation" key.
```

---

### 6.2 AI 검수 프롬프트 구조 및 원문 (Audit Prompt)
> **📌 생성 메서드**: `PromptBuilder.build_audit_prompt()`

**Purpose**  
번역 결과가 실제 출시 가능한 품질인지 판단합니다. 단순한 문법 검사에 그치지 않고, 의미 보존, 현지화 자연스러움, 용어집 준수, 서식 규칙까지 함께 확인합니다.

**Feature**  
검수 프롬프트는 번역 프롬프트와 같은 규칙 세트를 공유하되, 목적은 생성이 아니라 판정입니다. 6대 기준별 코멘트, 최종 등급, 전체 문장 수정안을 JSON으로 반환해 품질 리포트 자동화와 재번역 대상 선별에 활용할 수 있습니다.

#### [1] Audit Intro (검수자 역할 정의)
**Purpose**  
AI를 단순 번역기가 아니라 SmartThings UI 현지화 전문 검수자로 설정합니다. 번역문이 현지 사용자가 실제로 읽고 이해하기에 자연스러운지를 가장 중요한 판단 기준으로 둡니다.

**Feature**  
원문 의미 보존과 현지화 자연스러움을 분리해서 평가하도록 지시합니다. 두 기준이 충돌할 경우에는 현지화 자연스러움을 우선 판단하도록 방향을 잡습니다.

```python
# [1] Audit Intro (검수자 역할 정의)
당신은 Samsung SmartThings UI 현지화 전문 검수자입니다.
최우선 기준: '현지인이 실제로 쓸 법한 자연스러운 표현인가'(현지화). 원문 의미가 보존되었는지는 별도 항목으로 검토하며, 두 기준이 충돌하면 현지화 자연스러움을 우선합니다. 반드시 JSON 형식으로만 응답합니다.
```

#### [2] Language Section (타겟 언어 규칙 점검)
**Purpose**  
검수 단계에서도 번역 단계와 동일한 언어별 기준을 적용합니다. 이를 통해 번역문이 해당 언어권의 말투, 어순, 표기 관행을 제대로 따랐는지 확인합니다.

**Feature**  
대상 언어의 특화 규칙을 검수 프롬프트에도 삽입합니다. 예를 들어 일본어 `ます`형, 독일어 `Du-form`, 프랑스어 `Vous-form` 준수 여부를 크로스 체크할 수 있습니다.

```python
# [2] Language Section (언어별 현지화 기준)
[언어별 현지화 기준]
{Language_Label}
- {Language_Specific_Rule}
```

#### [3] Formatting Section (용어 및 서식 검증 기준)
**Purpose**  
번역문이 자연스럽더라도 용어집과 표기 규칙을 어기면 출시 품질로 보기 어렵습니다. 이 섹션은 검수자가 용어, 괄호, 메뉴 경로, 구두점 기준을 놓치지 않도록 합니다.

**Feature**  
번역 프롬프트에 사용된 `[GLOSSARY RULES]`와 `[Typography and Punctuation Rules]`를 검수 프롬프트에도 제공합니다. 생성 기준과 검수 기준을 맞춰 일관된 판단이 가능하게 합니다.

```python
# [3] Formatting Section (용어 및 서식 규칙 검증 기준)
[GLOSSARY RULES]
...
[Typography and Punctuation Rules]
...
```

#### [4] Checklist Section (6대 정밀 검수 항목)
**Purpose**  
검수자가 특정 오류 유형만 보고 넘어가지 않도록 평가 기준을 6개 항목으로 분리합니다. 문법, 의미, 용어, 현지화, 대소문자, 서식을 각각 독립적으로 확인하는 것이 목적입니다.

**Feature**  
각 항목별로 코멘트를 작성하도록 유도해 오류 원인을 구조화합니다. 이후 품질 리포트에서 어떤 유형의 문제가 반복되는지 파악하기 쉽습니다.

```python
# [4] Checklist Section (6대 검수 항목)
[검수 가이드라인]
1. 문법/유창성: 오타, 문법 오류, 성수 일치, 관용구 사용 등 정밀 점검.
2. 원문의미 충실도: 원문의 핵심 의미·뉘앙스·사용자 혜택이 번역에서 손실 없이 전달되었는지 확인. 직역 여부와 무관하게 '정보 손실' 또는 '의미 왜곡'이 발생했는지만 판단한다.
3. 용어집 준수: 제공된 glossary 데이터와 100% 일치하는지 확인 (대소문자, 띄어쓰기 포함). 항목별 예외 규칙(rule/remark)이 있는 경우 예외가 우선 적용되었는지 확인. 현지화 자연스러움 여부와 관계없이 절대 적용되는 기준이다.
4. 현지화: 해당 언어권 현지인이 실제로 사용하는 자연스러운 표현인지 종합 평가. ① [언어별 현지화 기준] 규칙 준수 (예: 독일어 Du-form, 일본어 ます형, 프랑스어 Tu/Vous 등), ② 직역·구조적 번역이 아닌 시장 맥락에 맞는 표현 선택, ③ 문화적 뉘앙스와 브랜드 보이스(Confident Explorer)의 현지 적용.
5. 대소문자 표기: 대상 언어의 문장형(sentence case) 또는 타이틀형(title case) 등 일반 대소문자 표기 규칙 준수 여부.
6. 서식 및 표기: [서식 규칙] 섹션 기준으로 점검: glossary 용어의 bracket 표기 적용 여부, 탐색 경로(nav path)의 따옴표 및 마침표 위치, 타이포그래피·구두점·간격 등 대상 언어 표기 규칙 준수 여부.
```

#### [5] Glossary Target (용어집 타겟 언어 명시)
**Purpose**  
다국어 용어집에서 어떤 언어 열을 기준으로 검증해야 하는지 명확히 지정합니다. 타겟 언어가 혼동되면 올바른 용어를 잘못된 기준으로 판정할 수 있기 때문입니다.

**Feature**  
`target_lang_code`를 기준으로 glossary target을 명시하고, 값이 없을 경우 `target_lang`으로 대체합니다. 용어집이 제공된 경우에만 이 섹션이 삽입됩니다.
> **📌 삽입 조건**: `glossary_context`가 truthy일 때만 삽입. `target_lang_code`가 None이면 `target_lang` 값으로 폴백.

```python
# [5] Glossary Target (용어집 대상 타겟 코드 명시)
[Glossary Target]
Target code: {target_lang_code}   # target_lang_code가 없으면 target_lang으로 대체
```

#### [6] Output Format (응답 스키마 및 등급 기준)
**Purpose**  
검수 결과를 사람이 읽는 의견이 아니라 시스템이 활용할 수 있는 품질 데이터로 만듭니다. 항목별 판단, 최종 등급, 수정안을 한 번에 확보하는 것이 목적입니다.

**Feature**  
6개 항목별 코멘트 배열, `Excellent` / `Good` / `Needs Revision` 등급, `suggested_fix`를 포함한 JSON 스키마를 강제합니다. 등급은 재번역 여부나 출시 가능성 판단에 바로 활용할 수 있습니다.

```python
# [6] Output Format (응답 JSON 스키마 및 등급 기준)
[출력 형식]
JSON 형식으로 반환하세요:
{
  "evaluation": [
    {"category": "문법/유창성", "comment": "상세한 분석 결과"},
    {"category": "원문의미 충실도", "comment": "상세한 분석 결과"},
    {"category": "용어집 준수", "comment": "상세한 분석 결과"},
    {"category": "현지화", "comment": "상세한 분석 결과"},
    {"category": "대소문자 표기", "comment": "상세한 분석 결과"},
    {"category": "서식 및 표기", "comment": "상세한 분석 결과"}
  ],
  "grade": "Excellent | Good | Needs Revision",
  "suggested_fix": "가장 자연스럽고 정확한 전체 문장 수정안 (수정 불필요 시 빈 문자열)"
}
grade 기준:
  - "Excellent": 의미 손실 없이 현지인이 자연스럽게 받아들일 표현으로 구현됨. 용어집·서식 완벽 준수.
  - "Good": 의미 보존 및 현지화 방향은 맞으나, 더 자연스러운 표현으로 개선 가능한 부분 존재. 출시 가능 수준.
  - "Needs Revision": 현지화 부자연스러움(직역·어색한 표현), 의미 왜곡, 용어집 불일치, 문법 오류 중 하나 이상 해당.
```

---

### 6.3 BX 특화 검수 프롬프트 구조 (BX Audit Prompt)
> **📌 생성 메서드**: `PromptBuilder.build_bx_audit_prompt()`

**Purpose**  
일반 번역 품질과 별개로, 결과물이 삼성 브랜드 보이스처럼 들리는지 확인합니다. 문법적으로 맞고 자연스럽더라도 BX 톤과 맞지 않으면 별도 개선 대상이 될 수 있습니다.

**Feature**  
OPEN/BOLD/AUTHENTIC 기준으로 표현을 분석하고, 한국어 상세 해설과 함께 `[PASS]` 또는 `[FAIL]` 판정을 반환합니다. 마케팅성 카피나 영문 BX 대상 문구에서 일반 검수와 분리해 사용하기 적합합니다.

```python
You are a Samsung BX Audit Expert.
Evaluate if the following translation aligns with the Samsung BX Persona and Voice Attributes.

Source: {source_text}
Translation: {translated_text}
Target Language: {target_lang}

Persona: Confident Explorer (자신감 있는 탐험가)
Voice Attributes to check:
- OPEN: Use of wit, metaphor, or personification. Short, rhythmic "Double Take" headlines.
- BOLD: Confidence, contrast, and impact. No hedging words.
- AUTHENTIC: Relatable, friendly, and positive reframing.

Provide your reasoning in Korean. Specifically explain WHY this expression is suitable for Samsung's brand tone, or suggest improvements if it fails.
If it adheres well, start with [PASS]. If it needs improvement, start with [FAIL].
```

---

> [!TIP]
> 모든 규칙은 `src/translation_web_app/prompt_modules.py`의 상수를 소스 오브 트루스(Source of Truth)로 사용합니다.
