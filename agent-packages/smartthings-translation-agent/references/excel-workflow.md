# Excel 워크북 워크플로우

SmartThings 번역 워크북의 구조와 안전한 분석·수정 규약.

## 워크북 포맷 (스토리 콘텐츠)

`@translation_data/@excel/` 의 `(CX Center) SmartThings_2.0_Story_Contents_*.xlsx` 형식.

| 위치 | 의미 | 상수 (rag_db_builder.py) |
|------|------|---------------------------|
| `C5` | story_id (예: `story_001`) | `STORY_ID_CELL` |
| 행 7~28 | 콘텐츠 영역 | `CONTENT_ROW_START`, `CONTENT_ROW_END` |
| B열 (2) | section code (예: `//section_001_1`) | `SECTION_COL` |
| C열 (3) | 콘텐츠 텍스트 | `CONTENT_COL` |

각 워크북은 **언어별 시트**를 가진다(보통 25개): `KR(한국)`, `US(미국)`, `UK(영국)`, `AU(호주)`, `SG(싱가포르)`, `FR(프랑스)`, `BE(벨기에)`, `CA(캐나다)`, `DE(독일)`, `IT(이탈리아)`, `ES(스페인)`, `NL(네덜란드)`, `SE(스웨덴)`, `AE(아랍에메리트)`, `PT(포르투갈)`, `BR(브라질)`, `RU(러시아)`, `TR(터키)`, `CN(중국)`, `TW(대만)`, `JA(일본)`, `PL(폴란드)`, `VN(베트남)`, `TH(태국)`, `ID(인도네시아)`.

- **소스 시트**: `KR(한국)`(Group A), `US(미국)`(Group B). 나머지는 타겟.
- section code 종류: `//story_NNN_title`, `//story_NNN_description`, `//section_NNN_M`, `//section_NNN_M_description`, `//section_NNN_M_disclaimer` 등.

## 1단계: 분석 (읽기 전용)

```bash
python scripts/workbook_inspect.py <path.xlsx>                    # 전체 시트 요약
python scripts/workbook_inspect.py <path.xlsx> --sheet "JA(일본)"  # 특정 시트
python scripts/workbook_inspect.py <path.xlsx> --cell-range C7:C28 # 셀 범위 덤프
python scripts/workbook_inspect.py <path.xlsx> --json              # 파싱용 JSON
```

출력: 시트 목록, 언어 코드 매핑, 각 시트의 story_id와 채워진 행(섹션코드 + 내용 미리보기 + 길이).

**이 스크립트는 절대 파일을 수정하지 않는다.** `data_only=True`로 읽기만 한다.

## 2단계: 편집안 제시 → 승인 → 납품본 적용

1. **편집 후보 제시**: 어떤 시트/셀을 왜 고칠지, before/after를 사용자에게 보여준다. (아직 수정 안 함)
2. **명시적 승인 대기**: 사용자가 "그렇게 해줘"라고 승인할 때까지 적용하지 않는다.
3. **저수준 적용** (임시 확인용):
   ```bash
   python scripts/workbook_apply_edits.py <path.xlsx> <edits.json | inline-json>
   ```
4. **납품용 적용(권장/필수)**: 승인된 story 수정안은 `/st-story-apply`로 적용한다. delivery scope를 명시하면 복사본 생성, scope 전체 재하이라이트, 값 변경 검증, `.delivery.json` manifest 생성을 한 번에 처리한다.
5. **수동 하이라이트 재적용**: 예외적으로 저수준 적용을 사용한 경우에는 수정본을 납품본으로 안내하기 전, 아래 명령으로 이번 납품 언어 전체를 재하이라이트한다 (자세한 내용은 "Glossary rich text highlight" 절 참조):
   ```bash
   python scripts/workbook_highlight_glossary.py <path.xlsx> --include-source-sheets --cell-range C7:C28
   ```
   최신 glossary 사용 여부와 `KR(한국)`/`US(미국)` source sheet 포함 여부를 반드시 사용자에게 보고한다.

### edits JSON 형식

```json
[
  {"sheet": "JA(일본)", "cell": "C10", "new_value": "新しいテキスト"},
  {"sheet": "DE(독일)", "row": 11, "col": "C", "new_value": "Neuer Text"}
]
```
- 좌표는 `cell`(`"C10"`) 또는 `row`+`col`(`col`은 문자 `"C"`/숫자 `3`) 중 하나.
- **하나라도 오류면 전체 중단**(부분 적용 방지) → 파일 미생성.

### 안전 보장 (스크립트 내장)

- 원본 파일 **수정 안 함**. `<원본>_revised_<타임스탬프>.xlsx` 복사본 생성.
- `cell.value`만 갱신 → 셀 레벨 폰트/색/병합 등은 보존한다.
- 단, Excel rich text(셀 내부 일부 글자만 파란색인 glossary 하이라이트)는 보존을 보장하지 않는다. 셀 값을 편집한 뒤 자동 산출본을 납품본으로 쓸 경우, 아래 "Glossary rich text highlight" 절차로 전체 하이라이트를 재생성한다.
- atomic write: `.tmp` 저장 후 `os.replace()`.
- 변경 로그 `<...>_revised_<ts>.changes.json` 동시 생성 (old/new 값 포함).
- **납품본 판정:** `*_revised_*.xlsx`만으로는 납품할 수 없다. `story-apply`의 `final` 경로 또는 전체 delivery scope 재하이라이트와 값 검증을 마친 파일만 최종본으로 안내한다.

## Section-level coherence review (섹션 맥락 검토)

**배경**: 기존 앱은 셀 단위 병렬 번역이라, section의 **title**을 번역할 때 같은 section의 **description** 맥락을 놓칠 수 있다. 그 결과 title이 너무 일반적이거나 description의 핵심 혜택을 반영하지 못하는 경우가 생긴다. 이 skill은 section 단위로 title↔description 정합성을 검토해 이를 보완한다.

### 절차
1. **그룹 추출**: `workbook_inspect.py --sections` 로 story/section 단위 그룹을 얻는다.
   ```bash
   python scripts/workbook_inspect.py <story.xlsx> --sheet "JA(일본)" --sections --json
   ```
   각 그룹은 `title` / `description` / `disclaimer`(opt) / `button`(opt) 필드를 가진다. `is_empty_or_placeholder: true`(빈 값/`x`)인 필드는 검토 대상에서 제외한다.
2. **section 단위 검토**: 각 section의 `title`이 `description`과 정합한지 아래 기준으로 판단한다.
3. **제안 제시**: `response-patterns.md`의 "C-2. 섹션 타이틀–디스크립션 맥락 검토" 템플릿으로 셀 위치·현재 title·판단·이유·제안을 제시한다. **이 단계에서는 Excel을 수정하지 않는다.**
4. **승인 후 적용**: 사용자가 승인하면 기존 정책대로 `workbook_apply_edits.py`로만 수정본(복사본)을 만든다.

### 검토 기준
- title이 description의 **핵심 기능/혜택**을 반영하는가
- title이 너무 **일반적**이지 않은가 (그 section만의 차별점이 드러나는가)
- title과 description의 **톤이 충돌**하지 않는가
- BX 적용 대상이면 **Open/Bold/Authentic** 관점에서 title이 적절한가 (`rules-sources.md`의 `BX_STYLE_RULES` 참조)
- RAG 사례가 있으면(`rag_lookup.py`) 기존 title/description pairing과 **충돌하지 않는가**
- 지칭(this/it/your device 등)·부사("간단히"/"쉽게"/"just"/"simply" 등) 반복은 section 단위가 아니라 **story 전체 단위**로 검토한다 (→ `response-patterns.md` C-2 확장판)

> ⚠ 이 검토는 분석·제안 단계다. 원본 Excel은 절대 수정하지 않으며, 적용은 항상 사용자 승인 + `workbook_apply_edits.py`를 거친다.

## Glossary rich text highlight (앱 highlight_only)

앱 본체에는 Excel 셀 안의 **용어집 target term 글자 조각만** 파란색 rich text로 바꾸는 `highlight_only` 파이프라인이 있다. skill에서는 이 로직을 재구현하지 않고 `workbook_highlight_glossary.py` 래퍼로 호출한다.

```bash
python scripts/workbook_highlight_glossary.py <story.xlsx> --sheets "BR(브라질)"
python scripts/workbook_highlight_glossary.py <story.xlsx> --cell-range C7:C28 --json
python scripts/workbook_highlight_glossary.py <story.xlsx> --cell-range C7:C28 --include-source-sheets --json
python scripts/workbook_highlight_glossary.py <story.xlsx> --single-source --source-sheet "US(미국)" --sheets "BR(브라질),DE(독일)"
```

### 동작 방식

- 기본 용어집: app repo의 `runtime/glossary/latest_glossary.csv`
- 기본 범위: `C7:C28`
- 기본 source grouping: `KR(한국)` → `US(미국)`, `JA(일본)`, `CN(중국)`, `TW(대만)` / `US(미국)` → 그 외 타겟 시트
- `--include-source-sheets`: source sheet 자체도 하이라이트 대상에 포함한다. 전체 재하이라이트/납품본 복구 시 기본적으로 사용한다.
  - `KR(한국)` source group: `KR(한국)`, `US(미국)`, `CN(중국)`, `TW(대만)`, `JA(일본)`
  - `US(미국)` source group: `US(미국)`, 그 외 US-source 타겟
- 결과: 원본을 덮어쓰지 않고 `<원본>_highlighted_<타임스탬프>.xlsx` 생성
- 하이라이트 색상: 앱 구현 기준 파란색 `0000FF`

### 안전 규칙

- 사용자가 실제 파일 생성을 요청하거나 승인했을 때만 실행한다.
- source/target 시트 선택이 불명확하면 실행 전 확인한다.
- 셀 값을 편집한 뒤 하이라이트를 복구할 때는 최신 glossary 파일을 확인하고, 가능한 한 `--include-source-sheets --cell-range C7:C28` 로 전체 재하이라이트한다.
- 하이라이트는 기존 target 텍스트를 번역하거나 수정하지 않고 rich text만 적용한다.
- 용어집 불일치·괄호·대소문자 로그는 보고서 텍스트에 남지만, 최종 판단은 필요 시 별도 검수로 확인한다.
- `Safe`처럼 일반 단어가 용어집 term으로 오탐된 경우 원본 glossary DB를 바로 수정하지 않는다. 필요한 경우 임시 glossary CSV에서 문제 term만 제외해 재번역/재하이라이트하고, 그 결과가 임시 기준임을 보고한다.
- 최신 glossary가 아니면 `IKEA` 등 신규 용어가 하이라이트에서 빠질 수 있다. 이 경우 자동 하이라이트 파일을 최종본으로 안내하지 말고, 최신 glossary를 확보하거나 수동 반영용 텍스트를 제공한다.

## 검수 리포트 후 deterministic 패턴 점검

LLM/NotebookLM 검수 등급은 후보 신호다. `Needs Revision`만 추리면 같은 오류가 `Good` 항목에 남을 수 있으므로, 한 오류가 확인되면 전체 워크북에서 같은 패턴을 찾는다.

필수 점검 예:
- bracket 오삽입/누락: `[smartphone]`, `[Samsung]`, `[SmartThings]` 등 실제 glossary 예외 규칙과 대조
- **bracket 뒤 복수형만 붙는 패턴** (예: `[Routine]s`): 우선 의심 대상. 용어집 원문 자체가 이미 복수형(예: `[Manual routines]`)이면 그대로 유지하고, 아니면 bracket은 유지한 채 문장 구조로 복수를 처리한다("자연스러운 영어"로 bracket을 풀어 쓰는 것보다 용어집 bracket 유지가 우선)
- dict/JSON 래핑: `{'translation': ...}`, `{'translatio n': ...}` 같은 출력 파싱 실패
- 비정상 공백/분절: 태국어 등에서 단어 중간에 들어간 공백
- 용어집 오탐: 일반 형용사 `safe`가 제품명 `Safe`처럼 유지되는 사례
- source group 확산: `UK/AU/SG`, `FR/BE/CA`처럼 같은 source를 공유하는 형제 시트

브리핑에서는 각 항목을 `수정 필요`, `검수 false positive`, `추가 확인 필요`로 재분류한다.

## 번역·검수 파이프라인 (LLM, 옵트인)

워크북 전체를 실제로 번역(+검수)하거나 기존 번역을 일괄 검수하려면 앱 파이프라인을 호출한다.
**LLM 크레딧을 소모**하므로 소량·단건은 셀프 모드(`prompt_preview.py`, 크레딧 0)를 먼저 고려한다.
(→ `self-vs-pipeline.md`)

```bash
python scripts/workbook_translate.py <story.xlsx> --pipeline --sheets "DE(독일)"   # 번역(+검수)
python scripts/workbook_translate.py <story.xlsx> --pipeline --translate-only ...  # 번역만
python scripts/workbook_audit.py    <story.xlsx> --pipeline --sheets "DE(독일)"     # 검수 전용
```

`--pipeline` 없으면 스크립트가 거부하고 셀프 모드를 안내한다. 원본은 수정되지 않는다(앱이 새 파일 생성).

## 주의

- 검수 결과 워크북(rich text 하이라이트 포함)은 `checker_service.py`가 생성한다. 이 skill의 `apply_edits`는 단순 `cell.value` 치환용이며, rich text 하이라이트는 `workbook_highlight_glossary.py`를 사용한다.
- 시트명 변종(`FR(프랑스)` vs `FR (프랑스)`)이 보이면 데이터 정합성 문제로 보고한다 (`rag-workflow.md` 참조).
