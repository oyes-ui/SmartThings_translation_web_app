# Glossary import contract (Claude for Excel)

`excel-chatgpt/officejs-glossary-import.js`와 같은 계약이다. Claude for Excel이 임의 코드
실행 경로를 제공하면 그 Office.js 함수를 그대로 재사용해도 된다. 고정 도구만 제공하는
경우, 그 도구로 동일한 계약을 만족시켜야 한다.

## 입력 payload

사용자가 명시적으로 CSV를 가져올 때만, path-free payload로 전달한다. 로컬 CSV 경로나
secret은 절대 포함하지 않는다.

```json
{
  "locale": "es_CO",
  "version": "optional-user-label",
  "checksum": "sha256-or-equivalent",
  "terms": ["term1", "term2"]
}
```

- `locale`, `version`, `checksum`, `terms`(비어 있지 않은 배열)가 모두 있어야 한다.
- `terms`는 trim 후 중복 제거한다.

## 동작 계약

1. workbook에 `__ST_GLOSSARY` sheet/table이 **이미 있으면 중단**한다 — 기존 glossary를 덮어쓰지 않는다.
2. 새 sheet를 만들고 `term, locale, version, checksum` 헤더 + 각 term 행을 채운다.
3. table로 등록한 뒤 sheet를 hidden + protected로 설정한다.
4. 결과로 `imported`(건수), `locale`, `version`, `checksum`을 반환한다.

## 금지 사항

- 로컬 CSV 경로를 payload에 담거나 기록하지 않는다.
- 기존 `__ST_GLOSSARY` table을 덮어쓰거나 삭제하지 않는다.
- 이 import는 실제 Claude for Excel session에서 검증되기 전까지 PoC 준비물이다. glossary
  substring highlight의 납품 기준은 계속 Delivery Python(`workbook_highlight_glossary.py`)이다.
