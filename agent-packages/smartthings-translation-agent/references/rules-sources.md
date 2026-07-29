# 규칙 출처 (Rules Sources)

번역·검수 규칙을 물어볼 때 참조할 canonical source와 우선순위.

## 우선순위

1. **`src/translation_web_app/rules/`** — **언어별 규칙과 BX 규칙의 Single Source of Truth.**
   런타임이 실제로 읽는 Markdown 파일. 충돌 시 **이 디렉터리가 최종 기준**.
   - `rules/languages/{canonical_key}.md` — 언어별 규칙 (파일명 = canonical key)
   - `rules/bx_style.md` — Samsung BX 보이스
2. **`src/translation_web_app/prompt_modules.py`** — 그 **외 모든 규칙**의 Single Source of Truth
   (용어집 마커·대괄호 규칙, 타이포그래피, 검수 체크리스트/등급, 시트코드 alias).
   ⚠️ 이 파일의 `LANGUAGE_LOCALIZATION_RULES` / `BX_STYLE_RULES` / `LANGUAGE_RULE_LABELS`는
   **레거시이며 런타임에서 참조하지 않는다.** 마이그레이션 검증용으로 남아 있을 뿐이므로
   언어·BX 규칙을 물어볼 때 이 상수들을 인용하지 말 것.
3. **`docs/comprehensive_rules.md`** — 사람이 읽기 좋게 정리한 종합 문서. 설명·근거가 풍부.
4. **Obsidian 번역 규칙 노트** — *선택적 보조*. 없어도 동작해야 한다. 사용자가 명시적으로 "옵시디언 노트랑 비교해줘"라고 할 때만 참조. (자동 연결은 향후 wiki skill/RAG로 예정)

## 규칙 파일 형식

YAML front matter가 유일한 normative 콘텐츠이고, 본문(body)은 사람이 읽는 설명이며 앱이 파싱하지 않는다.

```yaml
schema_version: 1
kind: language
canonical_key: German        # 반드시 파일명 stem과 일치
display_name: German Du-form Consistency
rule_order: sequence         # 배열 순서 = 모델이 보는 순서
rules:
- rule_id: german-001
  scope: [app_prompt]        # app_prompt | agent_audit | excel_apply
  text: Use Du-form consistently unless ...
```

- `scope: [app_prompt]` — 앱 번역/검수 프롬프트에 들어간다.
- `scope: [agent_audit]` — **프롬프트에 들어가지 않는다.** 에이전트 검수 체크리스트 전용.
  코드에서는 `rules_loader.get_rules().rules_for_scope("agent_audit")`로 조회한다.
- 파일명에 공백이 있는 4개(`European Portuguese.md`, `Brazilian Portuguese.md`,
  `Simplified Chinese.md`, `Traditional Chinese.md`)는 셸에서 반드시 따옴표로 감쌀 것.
- 규칙 파일이 깨지면 앱이 기동 자체를 거부한다(fail-fast). 조용히 규칙이 빠지는 일은 없다.
- 규칙 수정은 **앱 재시작 후 반영**된다(프로세스 시작 시 1회 로드, hot-reload 없음).

## 상수 → 질문 매핑

| 출처 | 다루는 내용 | comprehensive_rules.md 섹션 |
|------|------------|------------------------------|
| `rules/languages/*.md` | 언어별 특화 규칙(존댓말, 따옴표, 철자 변형 등) | §3 |
| `rules/bx_style.md` | Samsung BX 브랜드 보이스(Confident Explorer) | §[4] BX Style |
| `COMMON_LOCALIZATION_STANDARD` | 모든 언어 공통 품질 기준(의도 보존, 직역 회피, 간결성) | §1, §[2] Common |
| `GLOSSARY_TERM_RULES` / `GLOSSARY_BRACKET_WRAP_RULE` | 용어집 용어 처리, 대괄호 래핑 | §4 |
| `GLOSSARY_EXEMPT_MARKERS` | 대괄호 생략 조건 (`no bracket`, `대괄호 제외`, `괄호 제외`) | §4 |
| `GLOSSARY_DISCLAIMER_NAV_EXCEPTION` 외 | 네비게이션 경로·disclaimer 예외 | §4 |
| `TYPOGRAPHY_AND_PUNCTUATION_RULES` | 구두점·간격·케이스 | §5 |
| `AUDIT_CHECKLIST_RULES` (6개) | 검수 6대 항목 | §2, §[4] Checklist |
| `AUDIT_GRADE_CRITERIA` | 등급 기준(Excellent / Good / Needs Revision) | §2 |

## 언어 키 형식 주의

- 언어 규칙 파일명(= canonical key)은 **풀워드**(`Japanese.md`, `German.md`, `French.md`).
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

> 정확한 문구는 항상 원본에서 직접 확인할 것 — 언어·BX 규칙은 `src/translation_web_app/rules/`,
> 나머지는 `prompt_modules.py`. 이 표는 탐색용 인덱스일 뿐이다.
