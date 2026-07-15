# 용어집 필터와 Obsidian 리포트 워크플로우

US 확정문구를 기준으로 다국어 검수 전 용어집 필터/활성화 후보를 판단하고, 필요하면 Obsidian용 Markdown 리포트로 남기는 절차.

## 핵심 원칙

1. **US 문구만 보고 일반 단어를 추측하지 않는다.** 반드시 실제 용어집(`latest_glossary.csv` 또는 사용자가 지정한 glossary CSV)과 매칭한 결과를 기준으로 제안한다.
2. **용어집의 현재 규칙을 함께 본다.** `비활성화`, `대괄호 제외`, 빈 규칙을 구분하고, 이미 비활성인 항목과 새로 판단해야 할 항목을 분리한다.
3. **대괄호 occurrence를 우선 신호로 본다.** US 확정문구에 `[Device control]`, `[Routine]`처럼 명시된 경우, 용어집에서 비활성화된 항목이라도 이번 파일/셀에서는 활성 후보로 검토한다.
4. **용어 단위보다 occurrence 단위로 판단한다.** 같은 용어라도 bracket이 있는 occurrence와 일반명사 occurrence를 분리한다.
5. **실제 로직 수정이 늦은 단계라면 운영 메모로 관리한다.** 용어집 값을 임시 활성화하고, 예외 occurrence는 수동 검수 메모에 남긴다.

## 입력 확인

작업 전 확인할 것:

- 대상 workbook 경로
- US 소스 시트 이름(보통 `US(미국)`)
- 검수 대상 언어 시트(예: `BR(브라질)`, `RU(러시아)`, `CN(중국)`)
- 사용할 용어집 CSV 경로
  - 기본 후보: app repo의 `runtime/glossary/latest_glossary.csv`
  - 사용자가 업로드/지정한 CSV가 있으면 그 파일 우선
- Obsidian 리포트 저장 경로가 필요한지

## 실제 매칭 절차

1. `workbook_inspect.py`로 US 시트의 콘텐츠 셀을 읽는다.
   ```bash
   python scripts/workbook_inspect.py <workbook.xlsx> --sheet "US(미국)" --cell-range C7:C28 --json
   ```
2. glossary CSV의 3행 헤더 구조를 확인한다.
   - key/source 컬럼
   - 규칙/설명 컬럼
   - `en_US` 또는 `영어_미국` 컬럼
3. US 텍스트와 glossary `en_US` 값을 실제 매칭한다.
   - 앱 로직과 맞추려면 `translation_web_app.checker_service.TranslationChecker.load_glossary_from_file(..., "en_US")`를 사용한다.
   - 단, bracket occurrence 판단은 별도로 `[ ... ]` 내부 문자열과 glossary term을 대조한다.
4. 결과는 셀별로 정리한다.
   - 용어집 항목
   - 걸린 셀
   - US 표현
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

US 원문의 bracket occurrence만 glossary와 대조:

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

US 원문에서 대괄호로 감싼 표현이 glossary와 일치하면 이번 파일/해당 셀에서 활성 후보로 본다.

예:

- `[Device control]` → `Device control` 활성 후보
- `[Routine]` → `Routine` 활성 후보
- `[Manual routines]` → `Manual routines` 활성 유지

### 수동 예외

같은 셀 또는 같은 파일에 bracket 없는 일반명사 occurrence가 있어도, bracket occurrence와 분리해서 본다.

예:

- `preferred routines`는 bracket 없는 일반 복수형이면 `Routine` 용어집 강제 대상으로 보지 않는다.
- `Device Control Panel`이 타이틀에 bracket 없이 쓰이면, US 원문 의도를 기준으로 수동 판단하고 bracket을 기계적으로 강제하지 않는다.

### 브랜드/제품명

`SmartThings`, `Galaxy`, `Samsung`처럼 `대괄호 제외` 규칙이 있는 항목은 유지한다. 이 항목들은 필터링 후보가 아니라 bracket 제외 준수 여부를 보는 대상이다.

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
US 원문의 bracket occurrence를 활성 기준으로 삼고 bracket 없는 일반명사는 수동 예외로 두는 것이 안전합니다.
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
## 3. US 확정문구 기준 실제 매칭
## 4. 공통 용어집 판단
## 5. 언어별 검수 섹션
## BR 포르투갈어(브라질)
## RU 러시아어
## CN 중국어(간체)
## 6. 다음 작업
## 7. 작업 메모
```

언어별 섹션은 차후 확장 가능하게 비워두되, 각 언어마다 `용어집 확인 대상` 표와 `셀별 메모` 자리를 둔다.

## Obsidian 저장 경로 주의

Obsidian vault/iCloud 경로는 workspace 밖일 수 있다. 저장 전 확인:

1. `ls -la <obsidian-report-dir>`로 읽기 가능 여부 확인
2. 실제 쓰기는 sandbox 밖이면 사용자 승인/권한이 필요할 수 있음
3. 작은 테스트 파일 생성/삭제로 쓰기 가능 여부를 확인할 수 있음
4. 쓰기 권한이 없으면 workspace 안에 초안을 만들고 경로를 사용자에게 알려준다

현재 사용자의 기본 리포트 경로로 확인된 위치:

```text
/Users/df_n67/Library/Mobile Documents/iCloud~md~obsidian/Documents/Dptls_vault_raw/10 Projects/SmartThings_2026/AI 번역검수 리포트
```

단, 이 경로는 환경 의존적이므로 다른 세션/사용자에게 일반화하지 않는다. 사용자가 경로를 지정하면 그 경로를 우선한다.

### Obsidian 저장 명령 예시

읽기 확인:

```bash
ls -la "$OBSIDIAN_REPORT_DIR"
```

쓰기 가능 여부 확인(작은 테스트 파일 생성 후 삭제):

```bash
touch "$OBSIDIAN_REPORT_DIR/.codex_write_test" && rm "$OBSIDIAN_REPORT_DIR/.codex_write_test"
```

초안 리포트를 workspace에 만든 뒤 Obsidian 폴더로 복사:

```bash
cp "output/260715-049_QuickPanel_BR_RU_CN_용어집필터_초안.md" \
  "$OBSIDIAN_REPORT_DIR/260715-049_QuickPanel_BR_RU_CN_용어집필터_초안.md"
```

복사 확인:

```bash
ls -l "$OBSIDIAN_REPORT_DIR/260715-049_QuickPanel_BR_RU_CN_용어집필터_초안.md"
```

## 안전 메모

- 이 워크플로우는 분석과 리포트 작성용이다.
- Excel 원본은 수정하지 않는다.
- 용어집 CSV를 실제로 수정해야 하면, 변경 전 사용자에게 `항목 / 현재 규칙 / 변경 규칙 / 이유`를 제시하고 승인받는다.
- LLM 재검수, RAG 재구축, 외부 NotebookLM 등록처럼 비용 또는 외부 연동이 있는 작업은 별도 승인 후 진행한다.
