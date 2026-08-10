# Claude for Excel Live Runbook

이 runbook은 custom Excel add-in을 배포하는 문서가 아니다. Claude for Excel의 연결된 live
session에서 SmartThings Excel Live skill을 실행할 때의 검증 순서다. `excel-chatgpt/live-runbook.md`와
단계 구성은 같지만, 1단계는 특정 실행 방식(코드 붙여넣기 등)을 전제하지 않는다 — Claude for
Excel이 실제로 제공하는 도구/기능만 신뢰한다(`claude-excel-capabilities.md` 참고).

## 0. 사전 조건

1. Microsoft Excel에 대상 workbook이 열려 있고 활성 상태다.
2. Claude for Excel(Anthropic) pane이 열려 있고 로그인돼 있다.
3. 연결된 Excel session이 현재 workbook을 대상으로 등록돼 있다.
4. 사용자가 정확한 sheet/range 또는 현재 selection 사용을 확인했다.

하나라도 없으면 쓰지 않는다. 원본 파일·임의 로컬 CSV 경로·`.env`를 찾아 읽지 않는다.

## 1. Read-only capability probe

`claude-excel-probe.md`의 체크리스트를 이 세션에서 실제로 확인한다. Claude for Excel이 코드
실행 경로를 제공하면 그 경로로, 고정 read/write 도구만 제공하면 그 도구로 다음을 확인한다.

- workbook 활성 sheet, selection, 보호 상태
- 대상 셀의 현재 text/value/formula
- 아직 아무 것도 쓰지 않았음(읽기 전용) — `decision: preview_only`로 취급

이 확인은 의도적으로 병합 셀 판정이나 부분 rich text 지원을 추측하지 않는다. 해당 상태는
실제 live 기능이 제공하는 읽기 결과로 별도 확인해야 한다.

## 2. Preview contract

변경 제안은 다음 최소 JSON으로만 만든다. `before`가 현재값과 일치하지 않으면 stale로 중단한다.

```json
{
  "surface": "claude_excel",
  "mode": "preview",
  "changes": [{
    "sheet": "CO(콜롬비아)",
    "cell": "C10",
    "before": "현재 값",
    "after": "제안 값",
    "reason": "es_CO 기본 tú 톤"
  }]
}
```

수식·보호/숨김 시트·병합 범위는 `blocked`로 표시한다. preview에는 write 기능을 호출하지 않는다.

## 3. 승인 후 live draft

사용자가 preview를 승인하고 session이 range 쓰기와 동등한 기능을 명시적으로 광고한 경우에만
승인 셀을 수정한다. 수정 직후 같은 대상 range를 다시 읽어 `after`와 수식 오류를 확인한다. 허용 범위 밖
변경이나 stale 값이 발견되면 결과 등급은 `draft`가 아니라 `blocked`다.

## 4. 하이라이트의 확정 경계

Claude for Excel의 쓰기 기능이 SmartThings workbook에서 substring rich text의 읽기·쓰기·기존 run
보존을 모두 입증하기 전에는 glossary 하이라이트를 live 적용하지 않는다. 셀 전체 강조로 자동
대체하지 않는다.

- 사용자가 셀 단위 강조를 명시: live draft로 제한 가능
- 용어 단위 색상 하이라이트 또는 납품본: Delivery Python `workbook_highlight_glossary.py`

## 5. Glossary import 준비

Delivery/CLI 환경에서는 `glossary_live_prepare.py <csv> --locale es_CO --version <label> --output payload.json`으로
사용자가 선택한 CSV를 path-free payload로 만든다. live session에서는 `claude-glossary-import.md`에
정리된 계약대로 그 payload를 전달해 새 `__ST_GLOSSARY` hidden/protected table을 만든다. 기존
표는 절대 덮어쓰지 않는다. 이 import 자체는 실제 Claude for Excel session에서 검증하기 전까지
PoC 준비물이며, glossary substring highlight의 납품 기준은 계속 Delivery Python이다.

## 6. 기록

`manifest-schema.md` 형식에 surface/mode/변경/검증 결과와 glossary의 locale/version/checksum만 기록한다.
파일 경로·CSV 원본 경로·secret은 남기지 않는다.
