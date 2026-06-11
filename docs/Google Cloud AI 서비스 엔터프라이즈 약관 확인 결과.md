# Google Cloud AI 서비스 엔터프라이즈 도입 검토 — 법무/약관 확인 결과

**조사 기준일:** 2026년 6월 9일  
**출처 문서:**
- Google Cloud Service Specific Terms (최종 수정: 2026.6.8)
- Google Cloud Platform Terms of Service
- Google Cloud Generative AI Indemnified Services (최종 수정: 2026.4.21)

---

## 1. 라이선스 및 과금 정책

| 항목 | 내용 | 상태 |
|------|------|------|
| 1-1 | 엔터프라이즈 최소 계약 수량(MOQ) | ✅ 확인완료 |
| 1-2 | 라이선스 가견적 | ✅ 확인완료 |

---

## 2. 법무 및 면책권 정책

### 2-1. 면책권 지원 AI 모델 리스트

**상태: ✅ 확인완료**

**근거 문서:** [Google Cloud Generative AI Indemnified Services](https://cloud.google.com/terms/generative-ai-indemnified-services)  
**관련 조항:** Service Specific Terms — Section 20.i (Additional Google Indemnification Obligations)

#### 핵심 내용
Google Cloud는 면책권(Indemnification)이 적용되는 서비스와 모델 목록을 공식 문서에 별도 페이지로 명시하고 있음.

#### 면책권 적용 대상 서비스 및 모델 (현행 목록)

**Google Cloud Platform (Vertex AI / Gemini Enterprise Agent Platform 계열)**
- Gemini Enterprise Agent Platform API (구 Vertex AI API)
  - 적용 모델: **Codey, Gemini, Imagen, PaLM, Veo** (GA 버전 한정)
- Agent Conversation on Gemini Enterprise Agent Platform (구 Vertex AI Conversation)
- Agent Search on Gemini Enterprise Agent Platform (구 Vertex AI Search)
- Grounding with Google Search
- Web Grounding for Enterprise
- Grounding with Google Maps
- Conversational Navigation (Automotive AI Agent)
- Gemini Enterprise
- NotebookLM Enterprise

**Google Workspace**
- Gemini in Workspace (구 Gemini for Google Workspace)
- Google Vids

#### 면책권 적용 범위 (2가지)
1. **Generated Output 면책**: 위 서비스의 수정되지 않은(unmodified) 산출물이 제3자의 지식재산권을 침해한다는 주장에 대해 Google이 방어 및 면책
2. **Training Data 면책**: Google이 사전 학습 모델 생성에 사용한 훈련 데이터가 제3자 지식재산권을 침해한다는 주장에 대해서도 Google이 방어 및 면책

#### 면책권 적용 제외 조건
아래의 경우 면책권 미적용:
1. 침해 가능성을 알면서도 해당 산출물을 생성·사용한 경우
2. Google이 제공하는 출처 인용, 필터, 지침 등 안전 도구를 무시·비활성화·우회한 경우
3. 권리자로부터 침해 통지를 받은 후에도 산출물을 계속 사용한 경우
4. 상표권 관련 분쟁으로 Customer가 해당 산출물을 상거래에 사용하는 경우
5. Customer가 Modified Model 또는 Customer Adapter Model 커스터마이징에 사용한 데이터에 필요한 권리를 보유하지 않은 경우
6. **무료(free of charge)로 제공되는 서비스 사용의 경우** → 유료 계약 시에만 면책 적용

---

### 2-2. 클라이언트 면책권 승계 (Pass-through) 가능 여부

**상태: ❌ 표준 약관 미지원 — 추가 협의 필요**

**근거 문서:** Google Cloud Platform Terms of Service  
**관련 조항:** Section 13 (Indemnification) — Section 13.1, 13.3

#### 핵심 내용
- Google의 면책권은 **Google ↔ Customer(계약 당사자) 및 그 계열사(Affiliates)** 범위에만 적용됨
- Customer가 산출물을 납품한 **독립적인 최종 클라이언트(End Client)에 대한 Pass-through 조항은 표준 약관에 존재하지 않음**
- Section 13.3에 따라 제3자 자료와 결합 시 면책 제외 조항도 적용됨

#### 결론
클라이언트 납품 후 저작권 분쟁이 발생할 경우, Customer가 자신의 클라이언트를 보호하려면:
- **Google과의 별도 Enterprise 계약 협상** 또는
- **추가 계약 조항(Custom Agreement) 삽입** 필요

→ Google 영업팀을 통한 별도 계약 조건 협의 권고

---

## 3. 출처 검증 기능 및 AI 재학습 방지

### 3-1. 입력 데이터 및 산출물 AI 모델 재학습 차단(Opt-out) 여부

**상태: ✅ 확인완료**

**근거 문서:** Google Cloud Service Specific Terms  
**관련 조항:** Section 18 (Training Restriction)

#### 핵심 내용

> *"Google will not use Customer Data to train or fine-tune any AI/ML models without Customer's prior permission or instruction."*

- Opt-out 신청 방식이 아닌, **고객의 사전 허락 없이는 학습 사용이 기본적으로 금지(Default Off)**
- 계약서(Service Specific Terms)에 명시적으로 보장됨
- Customer가 명시적으로 허용하거나 지시한 경우에 한해서만 학습 사용 가능

---

### 3-2. 산출물 출처 증명 시스템 제공 여부

**상태: ⚠️ 일부 확인 — 추가 확인 필요**

**근거 문서:** Google Cloud Service Specific Terms (해당 조항 없음)

#### 항목별 현황

| 수단 | 약관 명시 여부 | 실제 제공 여부 | 비고 |
|------|--------------|--------------|------|
| 생성 Audit Log | ❌ 약관 미명시 | △ 기능 제공 | Cloud Logging을 통해 API 호출 기록 관리 가능하나, 계약상 보장 아님 |
| 비가시적 워터마크 | ❌ 약관 미명시 | △ 일부 제공 | SynthID(텍스트/이미지 워터마킹) 기술 존재, 모든 모델 적용 여부 별도 확인 필요 |
| C2PA (Content Credentials) | ❌ 약관 미명시 | ❌ 미확인 | GCP 약관 내 C2PA 표준 관련 언급 없음, 별도 확인 필요 |

#### 결론
- 약관 문서상으로는 출처 증명 시스템에 대한 보장 조항이 **존재하지 않음**
- 실제 기술 기능(SynthID 등)은 별도 제품 문서(Google DeepMind, Vertex AI 기능 명세)에서 추가 확인 필요
- 엔터프라이즈 계약 시 **계약서 내 명시적 보장 요청** 권고

---

## 4. 슬라이드 반영용 비용/상업/저작권 요약

### 4-1. 비용 산정 기준

**상태: ✅ 산정완료**

**기준 서비스:** Google AI Studio API key / Gemini API paid tier  
**환율 기준:** 1 USD = 1,500원 (슬라이드 계산 편의 기준)  
**산정 범위:** AI 번역/검수 + 이미지 생성 + 영상 생성

> 비용 산정은 Google AI Studio/Gemini API 공개 단가 기준입니다. Gemini Enterprise Agent Platform 적용 시 Agent Runtime, 저장소, Vector Search, Pipeline, Notebook, 관리 비용 및 Enterprise 견적 조건에 따라 실제 비용이 달라질 수 있습니다.

#### AI 번역 및 검수

- 과금 방식: **토큰 사용량 기반 후불 과금**
- 대표 산식: **1,866원 x 월 6개 x 8회 = 약 89,568원**
- 보조 근거:
  - `docs/token_cost_report.md`의 RAG ON 셀 단위 산정 기준
  - Story 015 추산: 약 1,032원
  - Story 006 실측: 약 2,742원
- 실제 비용은 텍스트량, 언어 수, RAG 사용량, 검수 output 길이에 따라 변동 가능

#### 이미지 생성

- 기준 모델: **Nano Banana Pro (`gemini-3-pro-image`)**
- 선택 사유: Nano Banana Pro와 Nano Banana 2 중 공식 단가가 더 높은 모델 기준으로 보수 산정
- 1K/2K 기준:
  - **$0.134/장 x 100장 = $13.40**
  - 원화 환산: **약 20,100원**
- 4K 기준:
  - **$0.24/장 x 100장 = $24.00**
  - 원화 환산: **약 36,000원**
- 슬라이드 표기: **100장 기준 약 2.0만~3.6만 원**

#### 영상 생성

- 기준 모델: **Omni 대체 산정용 Veo 3.1 Standard**
- 선택 사유: Omni API 공식 단가가 확인되지 않아, 공식 Gemini API 가격이 공개된 Veo 3.1 Standard를 proxy로 사용
- 720p/1080p 기준:
  - **$0.40/sec x 10초 x 1개 = $4.00**
  - 원화 환산: **약 6,000원**
- 4K 기준:
  - **$0.60/sec x 10초 x 1개 = $6.00**
  - 원화 환산: **약 9,000원**
- 슬라이드 표기: **10초 1개 기준 약 0.6만~0.9만 원**
- 주석: Omni 공식 API 가격이 공개되면 Veo 3.1 proxy 단가를 Omni 단가로 교체 필요

#### 총 예상 비용

| 구분 | 번역/검수 | 이미지 | 영상 | 합계 |
|------|----------:|-------:|-----:|-----:|
| 기본형 | 89,568원 | 20,100원 | 6,000원 | **약 115,668원** |
| 4K 최대형 | 89,568원 | 36,000원 | 9,000원 | **약 134,568원** |

**슬라이드 표기:** 총 약 **11.6만~13.5만 원** 예상

---

### 4-2. 슬라이드 문구

#### Enterprise 요금제 비용

- Google AI Studio API key 기반 사용량 과금
- 번역/검수: 약 89,568원
- 이미지: Nano Banana Pro 기준 100장 약 2.0만~3.6만 원
- 영상: Omni 대체 Veo 3.1 기준 10초 1개 약 0.6만~0.9만 원
- 총 예상: 약 11.6만~13.5만 원
- 텍스트량, 언어 수, 이미지 해상도, 영상 길이에 따라 변동
- Agent Platform 적용 시 플랫폼 리소스 및 Enterprise 견적 조건에 따라 변동

#### 상업적 사용 가능 여부

- Google Cloud / Gemini API 유료 사용 시 업무·상업 목적 사용 가능
- Generated Output은 Customer Data로 취급
- Google은 Generated Output의 신규 IP 소유권을 주장하지 않음
- 단, 금지 사용 정책, 의료/미성년자 대상 제한, 경쟁 모델 개발 제한 등 약관 준수 필요

#### 저작권/보안 정책 분석

- 유료 Indemnified Services는 Generated Output 및 Training Data 관련 IP 면책 가능
- Gemini Enterprise Agent Platform API의 GA Gemini, Imagen, Veo 등은 면책 대상
- 무료 사용, 안전 도구 우회, 침해 가능성 인지 후 사용, 상표권 분쟁 등은 면책 제외
- Google은 고객 사전 허가/지시 없이 Customer Data를 모델 학습·파인튜닝에 사용하지 않음
- 최종 클라이언트 Pass-through 및 C2PA 보장은 별도 계약/기능 확인 필요

---

## 종합 요약

| 번호 | 항목 | 결과 | 비고 |
|------|------|------|------|
| 1-1 | 엔터프라이즈 최소 계약 수량(MOQ) | ✅ 확인완료 | |
| 1-2 | 라이선스 가견적 | ✅ 확인완료 | |
| 2-1 | 면책권 지원 AI 모델 리스트 | ✅ 확인완료 | 공식 목록 페이지 존재, 유료 한정 |
| 2-2 | 클라이언트 Pass-through 가능 여부 | ❌ 미지원 | 별도 Enterprise 계약 협상 필요 |
| 3-1 | AI 학습 재학습 차단(Opt-out) | ✅ 확인완료 | 계약서에 기본 금지 명시 |
| 3-2 | 산출물 출처 증명 시스템 | ⚠️ 추가 확인 필요 | 약관 미명시, 기능 별도 확인 필요 |
| 4-1 | 번역/이미지/영상 비용 산정 | ✅ 산정완료 | 총 약 11.6만~13.5만 원 예상 |
| 4-2 | 슬라이드 문구 | ✅ 작성완료 | 비용/상업적 사용/저작권·보안 3개 박스용 |

---

## 참고 링크

- [Google Cloud Service Specific Terms](https://cloud.google.com/terms/service-terms)
- [Google Cloud Platform Terms of Service](https://cloud.google.com/terms/)
- [Generative AI Indemnified Services 목록](https://cloud.google.com/terms/generative-ai-indemnified-services)
- [Google Cloud Data Processing Addendum](https://cloud.google.com/terms/data-processing-addendum)
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini Enterprise Agent Platform](https://cloud.google.com/products/gemini-enterprise-agent-platform)
