# 용어집 필터와 Obsidian 리포트 워크플로우

대상 언어의 source group 확정문구를 기준으로 다국어 검수 전 용어집 필터/활성화 후보를 판단하고, 필요하면 Obsidian용 Markdown 리포트로 남기는 절차.

## 핵심 원칙

1. **대상 언어의 source 문구만 보고 일반 단어를 추측하지 않는다.** 먼저 source group을 확정하고 실제 용어집(`latest_glossary.csv` 또는 사용자가 지정한 glossary CSV)과 매칭한 결과를 기준으로 제안한다.
2. **용어집의 현재 규칙을 함께 본다.** `비활성화`, `대괄호 제외`, 빈 규칙을 구분하고, 이미 비활성인 항목과 새로 판단해야 할 항목을 분리한다.
3. **대괄호 occurrence를 우선 신호로 본다.** source 확정문구에 `[Device control]`, `[Routine]`처럼 명시된 경우, 용어집에서 비활성화된 항목이라도 이번 파일/셀에서는 활성 후보로 검토한다.
4. **용어 단위보다 occurrence 단위로 판단한다.** 같은 용어라도 bracket이 있는 occurrence와 일반명사 occurrence를 분리한다.
5. **실제 로직 수정이 늦은 단계라면 운영 메모로 관리한다.** 용어집 값을 임시 활성화하고, 예외 occurrence는 수동 검수 메모에 남긴다.

## 입력 확인

작업 전 확인할 것:

- 대상 workbook 경로
- source group과 소스 시트 이름: KR source는 `KR(한국) → JA/CN/TW/US`, US source는 `US(미국) → BR/RU/DE…`
- 검수 대상 언어 시트(예: `BR(브라질)`, `RU(러시아)`, `CN(중국)`)
- 사용할 용어집 CSV 경로
  - 기본 후보: app repo의 `runtime/glossary/latest_glossary.csv`
  - 사용자가 업로드/지정한 CSV가 있으면 그 파일 우선
- Obsidian 리포트 저장 경로가 필요한지

## 실제 매칭 절차

1. 대상 언어의 source group을 확정하고 `workbook_inspect.py`로 그 source 시트 콘텐츠 셀을 읽는다.
   ```bash
   python scripts/workbook_inspect.py <workbook.xlsx> --sheet "US(미국)" --cell-range C7:C28 --json
   ```
2. glossary CSV의 3행 헤더 구조와 source group에 해당하는 source 열을 확인한다.
   - key/source 컬럼
   - 규칙/설명 컬럼
   - source group에 맞는 source 컬럼(`en_US`/`영어_미국` 또는 `ko_KR`/`한국어`)
3. 선택한 source 텍스트와 glossary의 해당 source 값을 실제 매칭한다.
   - 앱 로직과 맞추려면 `translation_web_app.checker_service.TranslationChecker.load_glossary_from_file(...)`에 source group에 맞는 언어 키를 사용한다.
   - 단, bracket occurrence 판단은 별도로 `[ ... ]` 내부 문자열과 glossary term을 대조한다.
4. 결과는 셀별로 정리한다.
   - 용어집 항목
   - 걸린 셀
   - source 표현
   - 현재 용어집 규칙
   - 이번 파일 판단

## 셸 명령어 예시

아래 명령은 skill 루트에서 실행한다. `APP_ROOT`, `WORKBOOK`, `GLOSSARY`는 작업 파일에 맞게 바꾼다.

```bash
APP_ROOT="/Users/df_n67/Documents/2_프로젝트/@SAMSUNG/@SmartThings_translation_web_app"
PY="$APP_ROOT/venv/bin/python"
WORKBOOK="/path/to/(CX Center) SmartThings_Story_Contents_049_QuicPanel_BR_RU_CN.xlsx"
GLOSSARY="$APP_ROOT/runtime/glossary/latest_glossary.csv"
```

아래는 US source 그룹 예시다. KR source 대상이면 `US(미국)`/`en_US`를 `KR(한국)`/해당 한국어 source 키로 바꾼다.

US 시트 요약:

```bash
"$PY" scripts/workbook_inspect.py "$WORKBOOK" --sheet "US(미국)" --json
```

US 콘텐츠 셀 원문 추출:

```bash
"$PY" scripts/workbook_inspect.py "$WORKBOOK" --sheet "US(미국)" --cell-range C7:C28 --json
```

용어집에서 주요 항목의 현재 규칙 확인:

```bash
rg -n '^Device control,|^Routine,|^Manual routines,|^Galaxy,|^SmartThings,' "$GLOSSARY"
```

앱의 실제 glossary loader 기준으로 US 문구와 glossary `en_US` 매칭:

```bash
cd "$APP_ROOT"
"$PY" - <<'PY'
import asyncio, json, sys
from pathlib import Path
from openpyxl import load_workbook

sys.path.insert(0, "src")
from translation_web_app.checker_service import TranslationChecker

workbook = Path("/path/to/workbook.xlsx")
glossary = Path("runtime/glossary/latest_glossary.csv")
sheet = "US(미국)"
rows = [7, 8, 10, 11, 15, 16]

async def main():
    checker = TranslationChecker()
    msg = await checker.load_glossary_from_file(str(glossary), "en_US")
    wb = load_workbook(workbook, data_only=True)
    ws = wb[sheet]
    out = {"load_message": msg, "matches": []}
    for row in rows:
        text = str(ws[f"C{row}"].value or "")
        terms = checker._get_relevant_glossary_terms(text)
        details = []
        for term in sorted(terms, key=str.lower):
            item = checker.glossary[term]
            details.append({
                "term": term,
                "rule": item.get("rule", ""),
                "target": checker._get_target_val(item.get("targets", {}), "en_US"),
            })
        out["matches"].append({"cell": f"C{row}", "text": text, "terms": details})
    print(json.dumps(out, ensure_ascii=False, indent=2))

asyncio.run(main())
PY
```

선택한 source 원문의 bracket occurrence만 glossary와 대조:

```bash
cd "$APP_ROOT"
"$PY" - <<'PY'
import csv, json, re
from pathlib import Path
from openpyxl import load_workbook

workbook = Path("/path/to/workbook.xlsx")
glossary = Path("runtime/glossary/latest_glossary.csv")
sheet = "US(미국)"
rows_to_check = [7, 8, 10, 11, 15, 16]

rows = list(csv.reader(glossary.open("r", encoding="utf-8-sig", newline="")))
headers = rows[:3]
width = max(len(row) for row in headers)
en_col = None
rule_col = None

for col in range(width):
    vals = [headers[row][col].strip() if col < len(headers[row]) else "" for row in range(3)]
    low = [value.lower() for value in vals]
    if any(value in ("en_us", "english_us") or value == "영어_미국" for value in low):
        en_col = col
    if any(("규칙" in value or "rule" in value or "설명" in value) for value in low):
        rule_col = col

terms = {}
for row in rows[3:]:
    if en_col is None or en_col >= len(row):
        continue
    term = row[en_col].strip()
    if not term or term.lower() == "lng":
        continue
    terms[term.lower()] = {
        "term": term,
        "source_key": row[0].strip() if row else "",
        "rule": row[rule_col].strip() if rule_col is not None and rule_col < len(row) else "",
    }

wb = load_workbook(workbook, data_only=True)
ws = wb[sheet]
result = []
for row in rows_to_check:
    text = str(ws[f"C{row}"].value or "")
    matches = []
    for bracketed in re.findall(r"\[([^\]]+)\]", text):
        matches.append({
            "bracketed_source": bracketed,
            "glossary_match": terms.get(bracketed.lower()),
        })
    result.append({"cell": f"C{row}", "text": text, "bracketed_matches": matches})

print(json.dumps(result, ensure_ascii=False, indent=2))
PY
```

## Bracket occurrence 판단 규칙

### 활성 후보

선택한 source 원문에서 대괄호로 감싼 표현이 glossary와 일치하면 이번 파일/해당 셀에서 활성 후보로 본다.

예:

- `[Device control]` → `Device control` 활성 후보
- `[Routine]` → `Routine` 활성 후보
- `[Manual routines]` → `Manual routines` 활성 유지

### 수동 예외

같은 셀 또는 같은 파일에 bracket 없는 일반명사 occurrence가 있어도, bracket occurrence와 분리해서 본다.

예:

- `preferred routines`는 bracket 없는 일반 복수형이면 `Routine` 용어집 강제 대상으로 보지 않는다.
- `Device Control Panel`이 타이틀에 bracket 없이 쓰이면, source 원문의 occurrence 의도를 기준으로 수동 판단하고 bracket을 기계적으로 강제하지 않는다.

### 브랜드/제품명

`SmartThings`, `Galaxy`, `Samsung`처럼 `대괄호 제외` 규칙이 있는 항목은 유지한다. 이 항목들은 필터링 후보가 아니라 bracket 제외 준수 여부를 보는 대상이다.

## 미적용(비활성) 후보 자동 추출

위 "수동 예외"를 기계가 먼저 훑어서 후보 목록으로 뽑는다. 판단은 여전히 사람이 한다.

```bash
python scripts/glossary_activation_candidates.py <워크북 또는 폴더> \
  --glossary <Glossary.csv> --app-root <app-root> --target-sheet "CO(콜롬비아)" \
  --output candidates.json --review-markdown candidates.md
```

탐지 규칙은 하나다: **용어집 키에 대문자가 있는데, 그 셀 안의 모든 출현이 소문자**. `The comfort of a
safe home`의 `safe`가 제품 용어 `Safe`로 매칭되던 경우가 이것이다. 한 번이라도 대문자로 쓰였으면
후보가 아니다 — activation manifest 키가 `(story, cell, source_term)`이라 셀 단위로 활성/비활성이
갈리기 때문이다.

**후보는 결정이 아니다.** 사람이 `confirmed: true`로 바꾼 항목만 매니페스트가 된다.

```bash
python scripts/glossary_activation_candidates.py x --glossary <Glossary.csv> \
  --target-sheet "CO(콜롬비아)" --from-candidates candidates.json \
  --emit-manifest inactive_manifest.json
```

이 매니페스트를 `agent_sheet_review.py --activation-manifest`로 넘기면 resolver가 해당 occurrence를
비활성으로 계산한다. 확인되지 않은 후보는 무시되므로, 검토 전에 실행해도 동작은 바뀌지 않는다.

왜 확인을 강제하는가: 실제로 쓰인 용어를 잘못 비활성화하면 필요한 번역이 조용히 빠진다. 반대 방향
(비활성 처리를 안 해서 사람 큐로 가는 것)보다 비싸다.

ES_CO 33개 워크북 실측: 후보 102건 / 용어 17종. 이걸 적용했을 때 검수 제안 21건 중 **정답인데 하드룰로
차단되던 5건이 0건**이 됐고, 납품 확정본이 모든 셀에서 게이트를 통과했다.

## 응답 형식

사용자에게 바로 답할 때는 아래 표를 우선 제공한다.

```markdown
| 용어집 항목 | 걸린 셀 | 현재 규칙 | 이번 판단 |
|---|---:|---|---|
| `Device control` | C8, C11, C16 | `비활성화` | bracket 명시 occurrence라 이번 파일에서는 활성 |
| `Routine` | C16 | `비활성화` | bracket 명시 occurrence라 이번 파일에서는 활성 |
| `Routine` | C11 | `비활성화` | `preferred routines`는 bracket 없음. 일반명사로 제외 |
```

그리고 결론을 짧게 덧붙인다.

```text
이번 파일에서는 용어 자체를 일괄 활성/비활성으로 자르기보다,
선택한 source 원문의 bracket occurrence를 활성 기준으로 삼고 bracket 없는 일반명사는 수동 예외로 두는 것이 안전합니다.
```

## Obsidian 리포트 작성

사용자가 Obsidian 리포트를 요청하면 Markdown 파일을 만든다. 권장 구성:

```markdown
---
title: "SmartThings Story {story_id} {asset} {locales} 용어집 필터 초안"
project: "SmartThings Translation"
story_id: "{story_id}"
asset: "{asset}"
target_locales:
  - BR
  - RU
  - CN
date: YYYY-MM-DD
status: "draft"
source_workbook: "{workbook filename}"
tags:
  - smartthings
  - localization
  - translation-review
  - glossary
  - obsidian-report
---

# {title}

## 1. 목적
## 2. 결론
## 3. Source group 확정문구 기준 실제 매칭
## 4. 공통 용어집 판단
## 5. 언어별 검수 섹션
## BR 포르투갈어(브라질)
## RU 러시아어
## CN 중국어(간체)
## 6. 다음 작업
## 7. 작업 메모
```

언어별 섹션은 새 리포트에서는 빈 틀로 두고, 기존 리포트 업데이트 시에는 해당 언어만 증분 갱신한다. 각 언어에는 아래 항목을 둔다.

```markdown
## {locale} {언어명}
- Source group: `{KR source | US source}` / `{source sheet}`
- AI 판정: `{요약}`
- 최종 판단: `{수정 필요 / 유지 / false positive / 추가 확인}`
- RAG 근거: `{exact/keyword/semantic 또는 사례 없음}`
- 반영 상태: `{미반영 / 사용자 수동 반영 / 승인된 복사본 반영}`

| 셀 | 현재/최종 문안 | 판단 | 근거 | 메모 |
|---|---|---|---|---|
```

## Obsidian 저장 경로 주의

Obsidian vault/iCloud 경로는 workspace 밖일 수 있다. 저장 전 확인:

1. `ls -la <obsidian-report-dir>`로 읽기 가능 여부 확인
2. 실제 쓰기는 sandbox 밖이면 사용자 승인/권한이 필요할 수 있음
3. 작은 테스트 파일 생성/삭제로 쓰기 가능 여부를 확인할 수 있음
4. 쓰기 권한이 없으면 workspace 안에 초안을 만들고 경로를 사용자에게 알려준다

vault 기본 경로는 코드나 skill에 고정하지 않는다. 사용자가 이번 작업의 vault 경로를 지정하면 그 경로를 우선한다.

초안은 workspace에 먼저 만든다. vault 발행, 기존 보고서의 locale 섹션 갱신, Base 현황판 생성은
명시 승인 후 `scripts/obsidian_workflow.py`의 `publish` 또는 `init-base`에 `--apply`를 붙여 실행한다.

### Obsidian 저장 명령 예시

읽기 확인:

```bash
ls -la "$OBSIDIAN_REPORT_DIR"
```

쓰기 가능 여부 확인(작은 테스트 파일 생성 후 삭제):

```bash
touch "$OBSIDIAN_REPORT_DIR/.codex_write_test" && rm "$OBSIDIAN_REPORT_DIR/.codex_write_test"
```

초안 리포트를 workspace에 만든 뒤 Obsidian 초안 형식으로 stage:

```bash
python scripts/obsidian_workflow.py stage "output/260715-049_QuickPanel_BR_RU_CN_용어집필터_초안.md" \
  --output "output/obsidian/260715-049_QuickPanel_BR_RU_CN_용어집필터_초안.md"
```

사용자 승인 후 vault에 발행:

```bash
python scripts/obsidian_workflow.py publish \
  "output/obsidian/260715-049_QuickPanel_BR_RU_CN_용어집필터_초안.md" \
  "$OBSIDIAN_REPORT_DIR" "260715-049_QuickPanel_BR_RU_CN_용어집필터_초안.md" --apply
```

## 안전 메모

- 이 워크플로우는 분석과 리포트 작성용이다.
- Excel 원본은 수정하지 않는다.
- 용어집 CSV를 실제로 수정해야 하면, 변경 전 사용자에게 `항목 / 현재 규칙 / 변경 규칙 / 이유`를 제시하고 승인받는다.
- LLM 재검수, RAG 재구축, 외부 NotebookLM 등록처럼 비용 또는 외부 연동이 있는 작업은 별도 승인 후 진행한다.
