# 규칙 출처 (Rules Sources)

번역·검수 규칙을 물어볼 때 참조할 canonical source와 우선순위.

## 우선순위

1. **`src/translation_web_app/prompt_modules.py`** — Single Source of Truth. 런타임 프롬프트가 실제로 사용하는 규칙 상수. 충돌 시 **이 파일이 최종 기준**.
2. **`docs/comprehensive_rules.md`** — 위 모듈을 사람이 읽기 좋게 정리한 종합 문서. 설명·근거가 풍부.
3. **Obsidian 번역 규칙 노트** — *선택적 보조*. 없어도 동작해야 한다. 사용자가 명시적으로 "옵시디언 노트랑 비교해줘"라고 할 때만 참조. (자동 연결은 향후 wiki skill/RAG로 예정)

## prompt_modules.py 주요 상수 → 질문 매핑

| 상수 | 다루는 내용 | comprehensive_rules.md 섹션 |
|------|------------|------------------------------|
| `COMMON_LOCALIZATION_STANDARD` | 모든 언어 공통 품질 기준(의도 보존, 직역 회피, 간결성) | §1, §[2] Common |
| `LANGUAGE_LOCALIZATION_RULES` | 언어별 특화 규칙(존댓말, 따옴표, 철자 변형 등) | §3 |
| `GLOSSARY_TERM_RULES` / `GLOSSARY_BRACKET_WRAP_RULE` | 용어집 용어 처리, 대괄호 래핑 | §4 |
| `GLOSSARY_EXEMPT_MARKERS` | 대괄호 생략 조건 (`no bracket`, `대괄호 제외`, `괄호 제외`) | §4 |
| `GLOSSARY_DISCLAIMER_NAV_EXCEPTION` 외 | 네비게이션 경로·disclaimer 예외 | §4 |
| `TYPOGRAPHY_AND_PUNCTUATION_RULES` | 구두점·간격·케이스 | §5 |
| `AUDIT_CHECKLIST_RULES` (6개) | 검수 6대 항목 | §2, §[4] Checklist |
| `AUDIT_GRADE_CRITERIA` | 등급 기준(Excellent / Good / Needs Revision) | §2 |
| `BX_STYLE_RULES` | Samsung BX 브랜드 보이스(Confident Explorer) | §[4] BX Style |

## 언어 키 형식 주의

- `LANGUAGE_LOCALIZATION_RULES`의 키는 **풀워드**(`"Japanese"`, `"German"`, `"French"`).
- RAG DB(`rag_pairs.target_lang`)와 Excel 시트명은 **코드 형식**(`"JA(일본)"`, `"DE(독일)"`).
- `PromptBuilder.get_language_rule(target_lang)`은 fuzzy substring 매칭으로 둘을 흡수한다.
- RAG 조회 시에는 `scripts/rag_lookup.py`가 코드/시트명/풀워드를 모두 받아 정규화한다 → `rag-workflow.md` 참조.

## 검수 6대 항목 (AUDIT_CHECKLIST_RULES)

검수 등급을 설명할 때 이 카테고리로 나눠 설명한다:
1. 문법·자연스러움 (Grammar/Fluency)
2. 의미 충실도 (Meaning Fidelity)
3. 용어집 준수 (Glossary Compliance)
4. 현지화 (Localization)
5. 케이스 (Case)
6. 서식 (Formatting/BX)

> 정확한 문구는 항상 `prompt_modules.py`에서 직접 확인할 것. 이 표는 탐색용 인덱스일 뿐이다.
