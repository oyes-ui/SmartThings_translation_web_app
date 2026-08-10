<!-- Adapted from smartthings-translation-agent/references/glossary-report-workflow.md (핵심 원칙 + Bracket occurrence 판단 규칙 섹션만). CLI/Obsidian 전용 절차는 이 skill에서 실행할 수 없어 제외했다. -->

# Glossary Activation Guide (노이즈 없이 glossary 적용하기)

`glossary-co-slim.md`의 `rule` 필드는 **용어 자체의 전역 스위치가 아니다.** `비활성화`라고
적힌 항목도 이번 셀의 source 원문에 대괄호로 명시돼 있으면 이번 occurrence에서는 활성
후보로 본다. 반대로 `rule`이 비어 있어도(=기본 활성) 무조건 강제 적용하지 않는다 — 아래
판단 절차를 거친다. **이 절차 없이 `rule` 값만 보고 일괄 적용/무시하면 오탐(노이즈)이
늘어난다.**

## rule 필드 해석

| `rule` 값 | 의미 |
| --- | --- |
| 빈 문자열 | 기본 활성. bracket 여부와 무관하게 term rule(대괄호 래핑 등)을 표준 적용 |
| `대괄호 제외` | 항상 적용 대상이지만 **절대 대괄호로 감싸지 않는다** (브랜드명 등) |
| `비활성화` | 기본은 비활성. **단, source 원문에 대괄호로 명시된 occurrence는 이번 파일/셀에서 예외적으로 활성 후보** |
| `대괄호 제외, 비활성화` | 위 두 규칙의 교집합: 활성화되더라도 대괄호는 절대 안 씀 |
| 기타(`고유명사, 원문 표기 유지` 등) | term별 remark를 그대로 따른다 — 일반화하지 않는다 |

## 판단 절차 (occurrence 단위, term 단위 아님)

1. **대괄호 occurrence를 우선 신호로 본다.** source 원문에 `[Device control]`처럼 명시돼
   있으면, glossary에서 `비활성화`인 항목이라도 이번 occurrence는 활성 후보로 본다.
2. **같은 셀에 bracket 있는 occurrence와 없는 occurrence가 섞여 있으면 분리해서 본다.**
   예: `[Routine]`은 활성 후보, 같은 텍스트의 `preferred routines`(bracket 없음, 복수형
   일반명사)는 강제 적용 대상으로 보지 않는다.
3. **브랜드/고유명사(`대괄호 제외`)는 필터링 후보가 아니다.** 활성 여부를 판단할 게
   아니라, 대괄호 없이 정확한 표기(`SmartThings`, `Galaxy`, `Samsung` 등)를 쓰고 있는지만
   확인한다.
4. **판단이 애매하면 후보로만 제시하고 자동 적용하지 않는다.** preview에 `현재 규칙`과
   `이번 판단(근거)`을 같이 보여줘 사용자가 최종 확인하게 한다.

## Preview에 넣을 표 형식

```markdown
| 용어집 항목 | 걸린 셀 | 현재 규칙 | 이번 판단 |
|---|---|---|---|
| `Device control` | C8 | `비활성화` | bracket 명시 occurrence라 이번 파일에서는 활성 |
| `Routine` | C11 | `비활성화` | `preferred routines`는 bracket 없음 — 일반명사로 제외 |
```

## 최신성 주의

`glossary-co-slim.md`은 `runtime/glossary/latest_glossary_260803.csv`에서 term/rule/
ko_KR/en_US/es_ES/es_CO 6개 열만 추출한 **2026-08-03 시점 정적 스냅샷**이다(원본은 26개
locale 컬럼을 갖고 있으나 CO 작업과 무관한 컬럼은 노이즈라 제외했다). glossary가 그 뒤
바뀌었을 수 있으니, 납품 직전이거나 term 활성 여부가 결과를 크게 바꾸는 경우 Delivery
Python 경로(`workbook_highlight_glossary.py`, 최신 `runtime/glossary/*.csv` 사용)로
재확인을 권한다. 워크북에 `__ST_GLOSSARY` table이 별도로 import돼 있으면 그쪽이 이 정적
스냅샷보다 우선한다(사용자가 명시적으로 넣은 최신 데이터이므로).
