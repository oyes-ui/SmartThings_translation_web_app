# Google Sheets 용어집 관리

## 기준 원본

- 기준 통합문서: [DF] Samsung STE 2.0 콘텐츠 다국어 처리
  - https://docs.google.com/spreadsheets/d/1kVEdSTqZcFHLK8tK6IsF3Jb5ks-42RQgDimZ-rziKxU/edit
- 기준 탭: `용어집 DB` (`gid=1476285275`)

`용어집 DB_신규용어 테스트` 같은 복제 탭은 검토·시험용이다. 사용자가 기준 탭을 변경하라고 명시하지 않은 한, 기준 탭을 직접 수정하지 않는다.

## 시트 읽기와 매핑

- 용어집의 1~3행은 헤더다. 원본 이력 파일의 `Language` 값은 **3행 `Lng`** 와 매칭한다. 2행의 locale code만으로 매핑하지 않는다.
- 동일한 `Lng`가 여러 열에 있으면 해당 값을 모든 일치 열에 반영한다. 예: `English_UK` 값은 두 `English_UK` 열에 반영한다.
- 이력에 없는 언어 값과 `Rule` 값은 추정하지 않는다. 빈값으로 두고 diff에 포함한다.
- 신규 용어를 테스트할 때는 기준 탭을 같은 통합문서 안에서 복제해 작업한다. 새 통합문서를 만들거나 기준 탭을 덮어쓰지 않는다.

## 내장 용어집 반영

"내장 용어집에 반영", "앱 용어집 업데이트", 또는 동등한 요청은 다음 흐름을 따른다.

1. 기준 탭과 현재 `GlossaryStore`를 읽어 신규·수정·삭제 후보를 구분한다.
2. source key, `Rule`, 영향 locale, 기존값→제안값을 간결히 보여주고 명시 승인을 받는다. 읽기·diff 작성만으로는 승인이 아니다.
3. 승인 후 기준 탭을 CSV로 내보내고 `scripts/glossary_manage.py import --mode merge --apply`로 반영한다. `replace`는 사용자가 명시적으로 요청한 경우에만 사용한다.
4. `status`/`list` 또는 export 재읽기로 반영 수와 주요 항목을 검증한다. 실패하거나 불일치하면 중단하고 상태를 보고한다.

현재 앱의 내장 용어집은 SQLite/CSV 기반이며 Google Sheets를 실시간으로 읽지 않는다. Google Sheets는 관리 원본이고, 내장 용어집 반영은 위 승인 기반 import를 통해서만 수행한다.
