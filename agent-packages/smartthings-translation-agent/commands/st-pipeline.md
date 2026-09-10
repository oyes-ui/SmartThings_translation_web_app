---
description: 승인된 API 번역·검수의 고급 호환 명령
argument-hint: translate|audit <대상과 요청>
---

`translate`는 `commands/st-translate.md`의 API 경로,
`audit`는 `commands/st-inspect.md`의 API 검수 경로에 위임한다.
실행 방식에 대한 기존 승인을 확인하고 비용이 드는 재시도에는 기존 크레딧 규칙을 적용한다.
API 실행을 local Excel 복구 루프에 넣지 않는다.
