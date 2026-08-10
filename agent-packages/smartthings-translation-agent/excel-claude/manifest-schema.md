# Live Excel Manifest Mapping (Claude for Excel)

이 레이어는 `report_format_spec.md`를 새로 대체하지 않는다. `excel-chatgpt/manifest-schema.md`와
필드 구조가 동일하며, `surface` 값만 다르다. 아래 필드는 live edit가 공통 report/manifest
계약으로 변환될 수 있도록 추가 기록하는 실행 메타데이터다.

```json
{
  "surface": "claude_excel",
  "mode": "preview | draft | delivery",
  "approval": "pending | approved",
  "glossary": {
    "locale": "es_CO",
    "version": "optional-user-label",
    "checksum": "sha256-or-equivalent"
  },
  "changes": [
    {
      "sheet": "CO(콜롬비아)",
      "cell": "C10",
      "before": "current value",
      "after": "approved value",
      "verification": "verified | blocked | fallback_delivery"
    }
  ]
}
```

원본 파일 경로, API 키, secret, 사용자가 선택한 CSV의 로컬 경로는 기록하지 않는다.

## 적용 게이트

`excel_live_manifest.py --require-approved`는 `approval: "approved"`이고 `mode`가 `draft` 또는
`delivery`이며 모든 변경이 `verified`일 때만 edits로 정규화한다. `preview`/`blocked`/`fallback_delivery`는
Delivery Python 적용에 전달되지 않는다. `surface`는 `chatgpt_excel`/`claude_excel`/`officejs_poc`
중 하나여야 하며, 나머지는 거부한다.
