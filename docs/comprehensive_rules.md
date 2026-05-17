# [가이드 02] 종합 규칙 모음 (Comprehensive Rules)

이 문서는 시스템 내부(`prompt_modules.py`)에 정의된 모든 번역 및 검수 규칙을 한곳에 모은 참조 문서입니다.

---

## 1. 공통 현지화 표준 (Common Standards)
모든 언어에 공통적으로 적용되는 기본 원칙입니다.
- 원문의 의도, 뉘앙스 및 사용자 혜택을 보존할 것.
- 직역보다는 해당 시장에 적합하고 자연스러운 표현을 우선할 것.
- 삼성 SmartThings 브랜드 톤(명확함, 자신감, 유익함)을 유지할 것.
- 문화적으로 어색한 관용구나 위험한 표현(공포 유발, 모호한 주장 등)을 피할 것.
- UI 문구는 간결하게 유지하되, 작업이나 혜택은 명확히 전달할 것.

---

## 2. 언어별 상세 특화 규칙 (Detailed Localization Rules)

시스템은 아래와 같이 각 언어 및 시장별로 세분화된 규칙을 적용합니다.

| 언어/지역 (Locale) | 상세 규칙 및 가이드라인 |
| :--- | :--- |
| **Korean** | 존댓말(honorific) 또는 문어체 스타일 일관성 유지. UI 전체 용어 일관성 확보. |
| **English (US)** | US 철자 사용 (color, personalize). 디스클레이머 마침표는 따옴표 밖(`".`)에 위치. |
| **English (UK)** | British 철자 사용 (colour, personalise, optimise). US 전용 어휘 지양. |
| **English (AU)** | British 철자 기반. 지나친 마케팅 톤을 지양하고 명확하고 유익한 톤 유지. |
| **English (SG)** | 국제 싱가포르 영어(British 철자) 사용. 현지 속어(Singlish) 사용 금지. |
| **German** | `Du-form` 기본 사용. 명사 대문자 표기 및 복합어 구조 준수. 기술적 어투 지양. |
| **Japanese** | `ます-form` 기본 사용. 자연스러운 UI 표현(操作/設定 등) 우선. 직역형 구조 지양. 내비게이션 경로는 `「 」` 사용 및 마침표를 안쪽에 배치. |
| **French** | `Tu/Vous` 중 하나를 일관되게 사용. 불필요한 대문자 사용 지양. |
| **French (BE/CA)** | **BE**: 벨기에 시장 톤 유지. **CA**: 북미 프랑스어 표준 및 자연스러운 어구 우선. |
| **Italian** | 자연스러운 이탈리아어 문장 구조 사용. 영어식 명사 나열 지양. |
| **Spanish** | `Usted` 기본 사용. 특정 요청이 없는 한 지역 중립적 스페인어 지향. |
| **Spanish (ES)** | 카스티야(Spain) 스페인어 사용. 라틴 아메리카 전용 어휘 지양. |
| **Dutch** | 직접적이고 간결한 문체. 영어식 어순이나 명사구 구조 지양. |
| **Swedish** | 간결한 UI copy. 영어식 Title Case 대신 Sentence Case(첫 글자만 대문자) 우선. |
| **Arabic** | 현대 표준 아랍어(MSA) 사용. UI 방향성(RTL), 구두점, 문장 끝맺음 규칙 준수. |
| **Portuguese (BR)** | 브라질 포르투갈어 전용 어휘 및 문구 사용. 유럽식 어휘/구조 지양. |
| **Portuguese (PT)** | 유럽 포르투갈어 전용 어휘 사용. 브라질식 어휘나 진행형(gerund) 구조 지양. |
| **Russian** | 러시아어 어순 준수. 영어식 명사구 직역 및 과도한 대문자 지양. |
| **Turkish** | 터키어 어순 준수. UI에 적합한 간결한 명령형 또는 묘사형 사용. |
| **Chinese (Simplified)** | 중국 본토 표준 용어 및 간체자 사용. 대만식 용어 지양. |
| **Chinese (Traditional)** | 대만 표준 용어 및 번체자 사용. 중국 본토식 용어 지양. |
| **Polish** | 격 변화(declension) 및 문법적 일치 준수. 영어식 명사 나열 지양. |
| **Vietnamese** | 자연스러운 어순 및 간결한 UI 표현. 영어식 대문자 패턴 지양. |
| **Thai** | 자연스러운 태국어 UI 표현. 불필요한 공백 및 구두점 복사 금지. |
| **Indonesian** | 간결하고 자연스러운 문체. 과하게 격식적인 구조나 영어식 직역 지양. |

---

## 3. 삼성 BX 스타일 가이드 (Samsung BX Style)
`BX Style Transcreation` 모드 활성화 시 적용되는 고차원 규칙입니다.

### 3.1 페르소나 (Persona)
- **자신감 있는 탐험가 (Confident Explorer)**: 두려움 없고(Fearless), 예리하며(Incisive), 진실되고(Real), 열린 마음(Open-minded)을 가진 브랜드 보이스.

### 3.2 핵심 보이스 속성 (Voice Attributes)
- **OPEN (창의적)**: 비유와 위트(Refined Wit) 사용, 기술의 의인화(Personify), 짧고 리듬감 있는 문답형 헤드라인(Double Take) 활용.
- **BOLD (대담한)**: 모호한 표현(hopefully, maybe) 제거, 대조(Contrast)를 통한 임팩트, 확고한 혁신의 가치 주장.
- **AUTHENTIC (진정성)**: 친구에게 말하듯 편안한 구어체, 부정적 단어 대신 긍정적 혜택으로 재구성(Positive Reframing).

### 3.3 부정 제약 사항 (Negative Constraints)
- 직역 금지.
- 부정적 프레이밍(Stress, Worry 등) 지양.
- 지나치게 격식적이거나 기술적인 어투 지양.
- 모호한 확신(hopefully, might 등) 금지.

---

## 4. 검수 및 채점 기준 (Audit Criteria)

AI 검수 시 다음 6가지 항목을 기준으로 점수를 매깁니다.

1. **문법/유창성**: 오타, 문법 오류, 성수 일치, 관용구 사용 등 정밀 점검.
2. **정확성 및 현지화 품질**: 원문의 의미와 뉘앙스가 충실히 보존되었는지, 현지인이 쓸 법한 자연스러운 표현인지 평가.
3. **용어집 준수**: 제공된 glossary 데이터와 100% 일치 여부 및 항목별 예외 규칙(rule/remark) 적용 확인.
4. **언어별 규칙 준수**: 각 언어별 현지화 기준 섹션에 명시된 규칙 준수 여부.
5. **대소문자 표기**: 타겟 언어의 일반적인 대소문자 표기 규칙(Sentence case 등) 준수 여부.
6. **서식 및 표기**: 용어집 브래킷(`[]`, `「」`) 적용, 탐색 경로의 따옴표 및 마침표 위치, 타이포그래피 준수 여부.

---

## 5. 타이포그래피 및 서식 규칙

- **Glossary Bracket**: `row_key` 문맥에 따라 용어집 단어를 감쌈. (Title/Button은 제외)
- **Nav Path**: 메뉴 경로는 이중 따옴표(일본어는 `「 」`) 사용. 
- **Punctuation Position (US English)**: `"Navigation path".` (마침표가 따옴표 밖)
- **Punctuation Position (International)**: `"Navigation path."` (마침표가 따옴표 안)
- **Punctuation Position (Japanese)**: `「Navigation path。」` (마침표가 `」` 안)

---

> [!TIP]
> 모든 규칙은 `prompt_modules.py`의 상수를 소스 오브 트루스(Source of Truth)로 사용합니다.
