# RAG 조회 워크플로우

`scripts/rag_lookup.py`는 **두 가지 모드**로 동작한다.

| 모드 | 키 | 의존성 | 기능 |
|------|----|--------|------|
| **offline** | ❌ 불필요 | SQLite(표준 라이브러리)만. chromadb/google-genai 미로드 | exact match, keyword(LIKE) 검색, story/section/언어 필터 |
| **semantic** | ✅ `GEMINI_API_KEY` | `translation_web_app` + chromadb + google-genai | 임베딩 유사도 검색 (진짜 RAG). exact+semantic 2단계 |

**모드 자동 결정** (`--mode auto`, 기본):
- `--story` / `--section` / `--keyword` 중 하나라도 있으면 → **offline** (메타데이터·키워드 조회는 임베딩 무관)
- `--query` 있고 키 있으면 → **semantic**
- 키 없으면 → **offline** (exact → keyword fallback)

`--mode offline` / `--mode semantic` 으로 강제 가능.

## 기본 사용

```bash
# auto: 키 있으면 semantic, 없으면 offline
python scripts/rag_lookup.py --query "<원문>" --target-lang <언어> [--source-lang English] [--n 3] [--json]

# offline 전용 기능
python scripts/rag_lookup.py --query "節約" --keyword --target-lang JA --source-lang Korean   # 부분일치
python scripts/rag_lookup.py --story story_001 --target-lang JA --source-lang Korean          # story 조회
python scripts/rag_lookup.py --section "//section_001_1"                                      # section 조회

# semantic 강제 (키 필요)
python scripts/rag_lookup.py --query "Save energy" --target-lang JA --mode semantic
```

- `--target-lang`: 코드(`JA`) / 시트명(`JA(일본)`) / 풀워드(`Japanese`) 모두 허용. **offline 에선 생략 가능**(모든 언어).
- `--source-lang`: 기본 `English`(→ US 그룹). `Korean`이면 KR 그룹. RAG DB는 두 그룹:
  - **KR 소스 그룹**: KR(한국) → JA/CN/TW/US
  - **US 소스 그룹**: US(미국) → DE/FR/ES/IT 등

## 한국어 리뷰용 조회

`KR(한국)`은 대부분 target 번역이 아니라 **source_text**로 저장되어 있다. 따라서 한국어 본문 리뷰에서 `--target-lang KR`을 지정하면 `target_lang=KR`을 찾지 않고, 자동으로 KR source group의 한국어 source 문장 기준으로 조회한다. API 키와 ChromaDB가 있으면 source-side **semantic** 조회를 수행하고, 키가 없거나 `--keyword`/`--story`/`--section`을 쓰면 source-side offline 조회로 동작한다.

```bash
python scripts/rag_lookup.py --query "스마트홈" --target-lang KR --keyword --json
```

출력의 `lookup_side: "source"`는 한국어 원문 사례 조회임을 뜻한다. 이때 `target_text`/`paired_target`은 해당 한국어 원문에 연결된 번역 참고 자료이며, 한국어 문체 판단의 1차 근거는 `korean_text`/`source_text`다.

## 키 없이 어디까지 되나 (offline)

RAG 데이터(`runtime/rag_db/rag_store.db`)만 있으면 **키 없이**:
- ✅ **exact match**: 원문 완전 일치 과거 번역
- ✅ **keyword 검색**: source/target 부분일치(LIKE)
- ✅ **story/section/언어 필터**: 메타데이터 기반 조회
- ❌ **유사도(semantic) 검색**: 쿼리 임베딩이 필요 → 키 필수

즉 "정확히 같은/포함하는 과거 사례 찾기"는 키 없이 가능하고, "비슷한 표현 찾기"만 키가 필요하다. `match_type` 필드로 `exact`/`keyword`/`lookup`/`semantic`을 구분 표기한다.

## 결과 해석

각 example은 다음 필드를 가진다:
- `match_type`: `"exact"`(원문 완전 일치, `similarity_score=1.0`) 또는 `"semantic"`(임베딩 유사, 코사인 거리<0.8)
- `similarity_score`: semantic일 때 `1 - 거리` (높을수록 유사)
- `target_lang`, `section_code`, `story_id`: 출처 메타데이터

**답변 시**: 근거가 exact match인지 semantic match인지 명시한다. semantic이면 "유사 사례(유사도 0.xx)"로 안내해 사용자가 신뢰도를 판단하게 한다.

## RAG 기반 표현 판단 우선순위

RAG 사례는 **정답이 아니라 과거 톤·용어 사용 경향의 근거**다. 표현 판단 시 아래 우선순위로 본다:

1. **용어집/기능명 bracket 강제 규칙** — `glossary_manage.py`/`rules-sources.md`의 `GLOSSARY_TERM_RULES` 등 명시적 규칙이 최우선. RAG 사례가 이와 다르면 규칙이 이긴다.
2. **문법·규칙 위반 여부** — 대상 언어 문법이나 `AUDIT_CHECKLIST_RULES` 위반이 있으면 RAG 톤 일치보다 우선 수정한다.
3. **RAG 톤 일치 vs BX 자연스러움** — 위 두 가지에 걸리지 않는 표현이면, RAG의 과거 톤 일치와 BX(`BX_STYLE_RULES`) 자연스러움은 **둘 다 참고 신호**이며 우열이 없다. 상충하면 상충 사실을 그대로 알리고 사용자가 선택하게 한다.

답변할 때는 판단 근거를 `RAG 기준` / `BX 기준` / `용어집 기준` 중 어디서 왔는지 표기한다 (→ `response-patterns.md` A·C-2 템플릿의 "근거" 필드).

## 표현 통일 조회 규약

표현을 story 안에서 통일할지 결정할 때는 단어만 검색하지 않는다. 아래 문맥을 함께 고정해 조회·해석한다.

1. **source group**: KR source(`KR → JA/CN/TW/US`) 또는 US source(`US → BR/RU/DE…`)를 먼저 확정하고 `--source-lang`을 맞춘다.
2. **대상 언어**: `--target-lang`으로 실제 검수 시트를 제한한다.
3. **콘텐츠 유형**: title, description, disclaimer/navigation을 구분한다. description 표현은 description 사례를 우선한다.
4. **판정 강도**: exact > keyword/동일 패턴 > semantic 순으로 가까운 사례임을 표시한다. 사례가 없으면 통일을 강제하지 않고 "RAG 근거 없음: 규칙/BX 기준의 권장"으로 보고한다.

disclaimer/navigation의 법적·UI 문체는 description의 일반 문장 선택 근거로 사용하지 않는다. 단, 동일한 UI 레이블 또는 용어집 표기를 확인하는 보조 근거로는 사용할 수 있다.

## ⚠ DB 데이터 품질 주의 (중요)

실제 `rag_pairs.target_lang`에는 **변종·오염 데이터**가 섞여 있다:

| 현상 | 예시 |
|------|------|
| 공백 변종 | `FR(프랑스)` 와 `FR (프랑스)` (괄호 앞 공백) |
| 오타 변종 | `PT(포르투갈)` 와 `PT(포루투갈)` |
| 파싱 쓰레기 | `046` 같은 비언어 값 |

`RagRetriever.retrieve()`는 `WHERE target_lang = ?` **정확 매칭**을 쓰므로, 단순히 `"FR(프랑스)"`로만 조회하면 `"FR (프랑스)"`에 저장된 사례를 **놓친다**.

**`rag_lookup.py`의 대응**: 입력 언어를 코드(`FR`)로 정규화한 뒤, DB에 실재하는 모든 변종 시트명을 찾아(`find_db_variants`) 각각 조회하고 병합한다. 출력의 `db_variants_matched` 필드로 어떤 변종을 조회했는지 확인할 수 있다. 변종이 2개 이상이면 `notes`에 경고가 표시된다.

> 근본 해결(DB 클린업)은 이 skill 범위 밖이다. 발견 시 사용자에게 데이터 정합성 문제로 보고한다.

## RAG 비활성 시 폴백

`retriever.is_available()`이 `False`인 경우(DB 미구축/비어 있음/로드 실패):
- `rag_lookup.py`는 `rag_available: false`와 안내 `notes`를 출력하고 정상 종료한다.
- 이때 **에이전트는 RAG 없이 규칙 기반으로 추론**하고, "RAG DB가 없어 과거 사례 없이 규칙만으로 판단했다"고 사용자에게 명시한다.
- DB 구축이 필요하면(크레딧 소모) 사용자 승인 후:
  ```bash
  PYTHONPATH=src python -m translation_web_app.rag_db_builder --build-all
  ```

## RAG DB 관리 CLI (`/st-ragdb`)

앱 `rag_db_builder` 는 이미 CLI다. 별도 래퍼 없이 app repo 루트에서 직접 실행한다. **status 만 크레딧 0**,
나머지 빌드/업데이트는 임베딩 크레딧을 소모하므로 실행 전 승인.

```bash
PYTHONPATH=src python -m translation_web_app.rag_db_builder --status            # 현황(크레딧 0)
PYTHONPATH=src python -m translation_web_app.rag_db_builder --pilot             # 첫 파일 1개만(테스트)
PYTHONPATH=src python -m translation_web_app.rag_db_builder --build-all [--force]
PYTHONPATH=src python -m translation_web_app.rag_db_builder --update-story story_025
```

## 언어 코드 ↔ 풀워드 매핑 (rag_lookup.py 내장)

| 풀워드 | 코드 | 풀워드 | 코드 |
|--------|------|--------|------|
| Korean | KR | Spanish | ES |
| English | US | Dutch | NL |
| German | DE | Swedish | SE |
| Japanese | JA | Arabic | AE |
| French | FR | Russian | RU |
| Italian | IT | Turkish | TR |
| Chinese | CN | Polish | PL |
| (Traditional) | TW | Vietnamese | VN |
| Portuguese | PT | Thai | TH |
| Portuguese_BR | BR | Indonesian | ID |

영어 변형(UK/AU/SG), 프랑스어 변형(BE/CA)도 코드로 직접 입력 가능.
