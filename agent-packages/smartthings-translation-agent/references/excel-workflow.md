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

## 2단계: 편집안 제시 → 승인 → 적용

1. **편집 후보 제시**: 어떤 시트/셀을 왜 고칠지, before/after를 사용자에게 보여준다. (아직 수정 안 함)
2. **명시적 승인 대기**: 사용자가 "그렇게 해줘"라고 승인할 때까지 적용하지 않는다.
3. **적용**:
   ```bash
   python scripts/workbook_apply_edits.py <path.xlsx> <edits.json | inline-json>
   ```

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
- `cell.value`만 갱신 → 폰트/색/병합 등 **서식 보존**.
- atomic write: `.tmp` 저장 후 `os.replace()`.
- 변경 로그 `<...>_revised_<ts>.changes.json` 동시 생성 (old/new 값 포함).

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

> ⚠ 이 검토는 분석·제안 단계다. 원본 Excel은 절대 수정하지 않으며, 적용은 항상 사용자 승인 + `workbook_apply_edits.py`를 거친다.

## Glossary rich text highlight (앱 highlight_only)

앱 본체에는 Excel 셀 안의 **용어집 target term 글자 조각만** 파란색 rich text로 바꾸는 `highlight_only` 파이프라인이 있다. skill에서는 이 로직을 재구현하지 않고 `workbook_highlight_glossary.py` 래퍼로 호출한다.

```bash
python scripts/workbook_highlight_glossary.py <story.xlsx> --sheets "BR(브라질)"
python scripts/workbook_highlight_glossary.py <story.xlsx> --cell-range C7:C28 --json
python scripts/workbook_highlight_glossary.py <story.xlsx> --single-source --source-sheet "US(미국)" --sheets "BR(브라질),DE(독일)"
```

### 동작 방식

- 기본 용어집: app repo의 `runtime/glossary/latest_glossary.csv`
- 기본 범위: `C7:C28`
- 기본 source grouping: `KR(한국)` → `US(미국)`, `JA(일본)`, `CN(중국)`, `TW(대만)` / `US(미국)` → 그 외 타겟 시트
- 결과: 원본을 덮어쓰지 않고 `<원본>_highlighted_<타임스탬프>.xlsx` 생성
- 하이라이트 색상: 앱 구현 기준 파란색 `0000FF`

### 안전 규칙

- 사용자가 실제 파일 생성을 요청하거나 승인했을 때만 실행한다.
- source/target 시트 선택이 불명확하면 실행 전 확인한다.
- 하이라이트는 기존 target 텍스트를 번역하거나 수정하지 않고 rich text만 적용한다.
- 용어집 불일치·괄호·대소문자 로그는 보고서 텍스트에 남지만, 최종 판단은 필요 시 별도 검수로 확인한다.

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
