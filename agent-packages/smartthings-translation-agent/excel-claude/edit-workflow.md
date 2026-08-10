# Live Excel Edit Workflow (Claude for Excel)

`excel-chatgpt/edit-workflow.md`와 안전 게이트·결과 등급이 동일하다. surface만 `claude_excel`로
바뀐다.

## 입력 계약

- 대상: workbook, sheet, range 또는 현재 selection
- 변경: `before`, `after`, 사유
- 옵션: `--highlight-glossary`, `--cell-highlight`(명시적 선택), `--delivery`

## 안전 게이트

1. 대상이 모호하면 수정하지 않는다.
2. 현재 값이 `before`와 다르면 stale proposal로 보고 중단한다.
3. 수식 셀, 병합 셀, 보호/숨김 시트는 기본 차단한다.
4. preview와 사용자 승인이 없으면 쓰기 작업을 하지 않는다.
5. 쓰기 뒤에는 값·수식·오류·허용 범위 diff를 재확인한다.

## 결과 등급

- `preview`: 읽기 전용 제안. 파일/워크북에 쓰지 않음.
- `draft`: 승인된 live 수정. rich text 또는 납품 범위 검증이 끝나지 않아 납품본이 아님.
- `delivery`: Delivery Python 경로로 glossary rich text, 지정 납품 범위, KR/US source sheet,
  텍스트 보존 검증까지 끝난 결과.
