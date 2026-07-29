# ChatGPT for Excel Live Runbook

이 runbook은 custom Excel add-in을 배포하는 문서가 아니다. ChatGPT for Excel의 연결된 live
session에서 SmartThings Excel Live skill을 실행할 때의 검증 순서다.

## 0. 사전 조건

1. Microsoft Excel에 대상 workbook이 열려 있고 활성 상태다.
2. Home 리본의 **ChatGPT**(OpenAI, LLC) pane이 열려 있고 로그인돼 있다.
3. 연결된 Excel session이 현재 workbook을 대상으로 등록돼 있다.
4. 사용자가 정확한 sheet/range 또는 현재 selection 사용을 확인했다.

하나라도 없으면 쓰지 않는다. 원본 파일·임의 로컬 CSV 경로·`.env`를 찾아 읽지 않는다.

## 1. Read-only capability probe

`officejs-probe.js`를 session의 Office.js 실행 경로에서 실행한다. 결과는 다음을 확인한다.

- workbook 활성 sheet, selection, 보호 상태
- 대상 셀의 현재 text/value/formula
- `writePerformed: false`, `decision: preview_only`

이 probe는 의도적으로 병합 셀 판정이나 부분 rich text API 지원을 추측하지 않는다. 해당 상태는
실제 live command가 제공하는 읽기 결과로 별도 확인해야 한다.

## 2. Preview contract

변경 제안은 다음 최소 JSON으로만 만든다. `before`가 현재값과 일치하지 않으면 stale로 중단한다.

```json
{
  "surface": "chatgpt_excel",
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

수식·보호/숨김 시트·병합 범위는 `blocked`로 표시한다. preview에는 write command를 호출하지 않는다.

## 3. 승인 후 live draft

사용자가 preview를 승인하고 session이 `write_range` 등 동등한 write 기능을 명시적으로 광고한 경우에만
승인 셀을 수정한다. 수정 직후 같은 대상 range를 다시 읽어 `after`와 수식 오류를 확인한다. 허용 범위 밖
변경이나 stale 값이 발견되면 결과 등급은 `draft`가 아니라 `blocked`다.

## 4. 하이라이트의 확정 경계

Office.js probe가 SmartThings workbook에서 substring rich text의 읽기·쓰기·기존 run 보존을 모두 입증하기
전에는 glossary 하이라이트를 live 적용하지 않는다. 셀 전체 강조로 자동 대체하지 않는다.

- 사용자가 셀 단위 강조를 명시: live draft로 제한 가능
- 용어 단위 색상 하이라이트 또는 납품본: Delivery Python `workbook_highlight_glossary.py`

## 5. Glossary import 준비

Delivery/CLI 환경에서는 `glossary_live_prepare.py <csv> --locale es_CO --version <label> --output payload.json`으로
사용자가 선택한 CSV를 path-free payload로 만든다. live session에서만 `officejs-glossary-import.js`에 그
payload를 전달해 새 `__ST_GLOSSARY` hidden/protected table을 만든다. 기존 표는 절대 덮어쓰지 않는다.
이 import 자체는 실제 Excel session에서 검증하기 전까지 PoC 준비물이며, glossary substring highlight의
납품 기준은 계속 Delivery Python이다.

## 6. 기록

`manifest-schema.md` 형식에 surface/mode/변경/검증 결과와 glossary의 locale/version/checksum만 기록한다.
파일 경로·CSV 원본 경로·secret은 남기지 않는다.
