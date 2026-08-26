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

## 0단계: 모든 Excel 쓰기 전 강제 변경 계약

셀 몇 개를 바꾸는 표준 도구뿐 아니라 작업용 스크립트를 새로 만드는 경우에도 이 단계를 생략할 수 없다.
시작 파일이 모든 항목의 정본이라고 가정하지 말고, 적용 전에 다음 세 역할을 파일·시트·범위 단위로
명시한다.

| 축 | 반드시 선언할 내용 | 예시 |
| --- | --- | --- |
| 값 정본 | 문안·수식·section code를 어디서 가져오는가 | 최신 글로벌 Fix, Local_adds |
| 구조 정본 | 시트 순서, 실제 행/열, section 순서·삭제를 어디에 맞추는가 | 지역향 원본 |
| 서식 정본 | 행 높이, 숨김, 테두리, 셀 스타일, rich-text run을 어디에서 보존/복사하는가 | 현재 Fix 또는 지역향 원본의 특정 행 |

한 파일이 여러 역할을 맡을 수는 있지만 역할을 생략할 수는 없다. 서로 다른 정본이 있으면 우선순위를
manifest에 기록한다. `변경 없음` 파일도 따로 열거하고 값·구조·서식의 논리 diff가 0인지 검증한다.

### 변경 유형을 먼저 분류한다

- **값 변경**: 셀 문안·수식만 바뀐다. 승인 좌표 외 값과 모든 구조/서식 diff는 0이어야 한다.
- **구조 변경**: 행/열/시트 추가·삭제·이동, section 재배치가 있다. 값 편집 도구로 대체하지 않는다.
- **서식 변경**: 행 높이, 숨김, 탭 색, 테두리, 셀 스타일, rich text가 바뀐다. 허용 범위를 별도로 기록한다.
- 둘 이상이면 각 축의 허용 diff를 따로 적는다. “전체 구조 보존” 같은 모호한 문구는 쓰지 않는다.

### 삭제 의미는 물리 구조로 판정한다

- 구조 정본에서 section이 사라졌다면 `cell.value = None`으로 비우지 말고 해당 **행 전체를 삭제**한다.
- section 행과 함께 구분 공백 행을 삭제할지, 마지막 공백 행을 남길지는 구조 정본의 실제 행과 맞춘다.
- 빈 셀도 안전하다고 가정하지 않는다. 삭제 행의 B/C 외 열에 있는 검토 메모·수식·주석과 스타일을
  사전 조사하고, 행과 함께 삭제되는 항목을 manifest에 기록한다.
- `max_row`와 비어 있지 않은 행만으로 통과시키지 않는다. 값 없는 스타일 셀, 행 높이, 숨김,
  outline/collapse, 병합 범위, 데이터 검증, freeze pane, 그림/차트 anchor도 구조 위험으로 본다.
- 구조 정본에 없는 서식 흔적이 남으면 실패다. 반대로 구조 정본에 있는 마지막 빈 구분 행을 임의로
  없애도 실패다.

### 생성과 검증을 분리한다

1. 전체 입력과 정본을 읽기 전용으로 snapshot하고 변경 계약 manifest를 만든다.
2. 빈 staging 폴더에 전체 배치를 생성한다. 사용자가 덮어쓰기를 허용해도 바로 최종 폴더에 쓰지 않는다.
3. 저장본을 `rich_text=True`, `data_only=False`로 다시 연다.
4. **생성기와 독립된 검증기**가 원본 정본에서 기대값을 다시 계산한다. 생성기가 만든 in-memory
   `expected` 객체를 그대로 재사용한 검증은 독립 검증으로 인정하지 않는다.
5. 배치 전체가 통과한 뒤에만 파일별 atomic replace로 최종 경로에 승격한다. 한 파일이라도 실패하면
   기존 최종 배치를 유지하고 실패 대상을 보고한다.

### 최종 통과 게이트

- **값**: 선언된 변경·삭제 좌표 외 값/수식 diff 0, source mapping과 Local_adds 셀 전수 확인
- **구조**: 구조 정본과 시트별 실제 행/열·section 순서·병합·숨김·행 높이 일치
- **서식**: 허용 범위 밖 셀/행 스타일 diff 0, 빈 구분 행 스타일 포함
- **rich text**: 표시 문자열뿐 아니라 run 순서·문자열·폰트 속성 보존, 위험 whitespace run 0
- **무변경 파일**: 값·수식·행 높이·셀 스타일의 논리 diff 0
- **round trip**: 저장 후 재개방 snapshot이 저장 직전 기대치와 일치
- **시각 확인**: 구조/서식 변경 파일과 위험 대표군을 렌더링해 잔여 테두리·과도한 빈 공간·잘림을 확인
- **기록**: manifest에 정본, 허용 값 변경, 행 삭제 범위, 삭제 행의 비콘텐츠 값, 검증 결과를 남긴다.

도구가 위 fingerprint를 제공하지 않으면 작업 전용 검증기를 만들어야 한다. 도구를 만들었다는 사실은
게이트 면제가 아니라 게이트를 자동화했다는 뜻이어야 한다.

### 새 일회용 도구의 코드 기반

새 `.xlsx` 수정 스크립트는 `scripts/workbook_mutation_guard.py`를 공용 기반으로 import한다. 최소한
다음 기능을 다시 작성하거나 복사본으로 분기하지 않는다.

- `clone_rich_value`: rich-text run을 평문으로 만들지 않는 값 복사
- `copy_row_layout`: 서로 다른 워크북 사이의 안전한 행 높이·셀 서식 복사
- `delete_rows_with_manifest`: B/C 밖 검토 메모와 서식까지 기록하는 실제 행 삭제
- `semantic_workbook_snapshot` / `snapshot_axis_diff`: 값·구조·서식·rich text·주석 독립 fingerprint
- `save_verified_atomic`: 임시 저장본을 재개방 검증한 뒤에만 원자적으로 교체

기능이 부족하면 일회용 스크립트에서 우회 구현하지 않는다. 공용 모듈을 먼저 확장하고 의미 있는
회귀 테스트를 추가한 뒤 그 API를 사용한다. 단순 복사·붙여넣기는 이후 수정이 도구마다 갈라지므로
공용 기반 사용으로 인정하지 않는다. 구조 변경 도구는 생성기와 별도 검증기를 계속 가져야 하며,
공용 snapshot은 검증 근거이지 생성기 자체 검증의 대체물이 아니다.

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

### Revision manifest와 검수 표시

워크북을 `/st-edit` 경로로 처음 받으면 원본 옆 `.st-history/<workbook-id>/baseline.json`에 기준선이
기록된다. 원본은 수정되지 않으며 기준선에는 원본 바이너리·값·구조 fingerprint만 저장한다. 승인된
복사본마다 revision manifest가 생성되어 변경 셀의 before/after, 문자 diff, 삭제 텍스트, 승인·검증
결과를 기록한다.

검수본이 필요하면 `workbook_incremental_highlight.py`를 사용한다. 수정/삽입 문자와 삭제 위치의 앞뒤
단어를 먼저 빨간 rich text로 표시하고, 이어 glossary term을 파란색으로 표시한다. 겹치는 구간은
glossary 파란색이 우선한다. 셀의 최종 텍스트 fingerprint와 glossary checksum이 같고 rich text가
유지된 셀은 재실행에서 건너뛴다. 납품본에는 빨간 표시를 남기지 않고 기존 delivery glossary 경로를
사용한다.

### 원어민 감수본 수용 → 최종안

감수 워크북이 `C=현재 문안 / F=감수 수정안 / H=감수 의견` 구조일 때는 `/st-story-apply`를 재사용하지 않는다. 검수 판단과 실제 반영을 분리한 `/st-review-apply`를 사용한다.

1. `/st-story-review` 및 감수안 대조로 언어별 `수용 / 부분 수용 / 유지`를 먼저 결정하고, 각 행을 `현재 → 감수안 → 최종안 → 쉬운 이유 → 원문/RAG 근거`로 리포트에 기록한다.
2. 결정만 담은 approval manifest를 만든다. `accept`는 F열을, `partial`은 명시한 `final_value`를, `hold`는 현재 C열을 유지한다. 실행 도구는 판정을 추론하지 않는다.
3. `workbook_review_apply.py`가 원본 불변 복사본에 승인된 `C7:C28`만 적용한다. 보호 언어에는 `accept`/`partial`을 허용하지 않는다.
4. 납품 템플릿에서 감수 메타데이터를 제거해야 할 때만 `--drop-review-columns E:H`를 명시한다. 삭제하지 않는 것이 기본값이다.
5. 생성한 1차 수용본을 기준으로 전 시트 `C7:C28`을 `--include-source-sheets` 방식으로 재하이라이트한다. 결과의 text preservation, 승인 목록=C열 diff, 보호 언어 값·수식·병합 구조를 모두 검증한 파일만 최종본으로 안내한다.

```bash
python scripts/workbook_review_apply.py <review.xlsx> <approval.json> \
  --output <story_1차수용_YYMMDD.xlsx> --glossary <Glossary.csv> \
  --drop-review-columns E:H --app-root <app_root> --json
```

결과 manifest에는 1차 수용본, 최종 하이라이트본, 변경 목록, 보호 언어 검증, C열 diff, highlight report 경로가 들어간다. Obsidian 리포트에는 이 경로와 검증 결과를 기존 내용을 지우지 않고 증분 기록한다.

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
