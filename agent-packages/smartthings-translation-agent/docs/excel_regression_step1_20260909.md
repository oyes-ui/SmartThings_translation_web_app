# Excel 회귀 자산 보존·분리 — 1단계 결과

작성일: 2026-09-09

## 범위와 결과

평가 문서를 최신화하고 사고 회귀 지식을 일회용 writer와 독립된 테스트로 보존했다.
프로덕션 코드, 기존 ignored 스크립트·테스트, 실제 업무 Excel은 변경하지 않았다.
새 실행 계약·상세 verifier·run 및 전체 시트 복사 기능은 다음 단계의 구현 범위다.

## 보존 자산

- `excel_regression_inventory_20260909.json`: 스크립트 52개·테스트 파일 4개의
  상대경로, 바이트 수, SHA-256, ignore 상태. 해시는 식별용이며 복구용 소스 백업이 아니다.
- `../tests/test_workbook_regression_invariants.py`: 자체 최소 워크북 빌더와 회귀 테스트 8개.
  ignored writer, 사용자 파일, 앱 API, 외부 서비스에 의존하지 않는다.
- 기존 일회용 파일은 그대로 유지했다. 신규 파일은 ignore되지 않으며 커밋 가능한 상태다.
  이 작업에서 git add 또는 commit은 수행하지 않았다.

## 사고 지식 대응

| 기존 사고·테스트 | 보존한 공용 불변식 | 한계·후속 작업 |
|---|---|---|
| rich-text replacement keeps fonts / crosses run boundary | clone·slice 조합의 문자별 폰트·공백 보존과 저장 후 재개방 | 제품명 탐색·치환 정책은 기존 writer에 유지 |
| cross-workbook style index | 서로 다른 스타일 테이블 사이 셀 스타일 의미 보존과 재개방 | 전체 시트·열 dimension 복사 primitive 이전은 미실시 |
| story 051 no-op | 수식·013·공백·5축 무변경 및 잘못된 layout 정본 사용 탐지 | contract의 정본 선택·no_op_files 정책은 다음 단계 |
| deleted regional sections | 빈 서식 꼬리 행 실제 삭제, E열 메모 기록, 마지막 구분 행 유지 | 국가·story별 삭제 범위 결정은 기존 writer에 유지 |
| 탭 색·행 높이·서식·주석·rich text 누락 | 각 사고 변형이 해당 snapshot 축의 차이로 나타나는지 검증 | 위치·속성 단위 상세 verifier는 다음 단계 |
| 검증 전 납품 | 검증 예외 시 새 최종 파일이 없고 임시 파일이 정리됨 | 기존 최종 파일 보존은 기존 guard 테스트가 담당 |
| English activation / US seed tests | 목록·해시와 기존 테스트 보존 | glossary 활성화·하이라이트 정책은 이번 mutation primitive 분리 범위 밖 |
| india 006 / story 047 | 목록·해시와 기존 테스트 보존 | 업무 고유 로직으로 ignored 원본 유지 |

## 테스트 파일의 3축 정본

실제 납품 파일을 만들지 않고 TemporaryDirectory 안에 합성 워크북만 생성한다.
값·구조·서식 정본은 각 테스트의 최소 입력과 명시된 기대 속성이다. 허용 변경은
각 테스트에 지정한 문자 삭제·행 삭제·서식 복사뿐이다. 저장 검증과 별도로 결과 파일을
다시 열어 literal 기대값·폰트·행 높이·서식 흔적·수식을 확인한다.
고의 오류 주입 테스트는 탐지기를 검증하며 결과를 납품하지 않는다.

## 검증

- 변경 전 로컬 전체: **190 passed, 16 subtests passed**.
- 변경 후 로컬 전체: **198 passed, 22 subtests passed**. 외부 의존성 deprecation 경고 2건.
- 신규 회귀 테스트: **8 passed, 6 subtests passed**.
- 격리 재현: 임시 폴더에 공용 guard와 신규 테스트 파일 두 개만 복사하여
  **8 passed, 6 subtests passed**. ignored writer 없이 실행됨을 확인했다.
  기존 프로젝트 venv를 사용한 코드 의존성 격리이며, 새 환경 설치 검증은 아니다.
- 전체 실행 명령: `../../venv/bin/python -m pytest tests/ -q -p no:cacheprovider`.
  시스템 Python에는 pytest가 없고 `.venv` 실행 경로는 유효하지 않아 기존 `venv`를 사용했다.

## 다음 단계

실행 contract 스키마와 상세 verifier를 정의한다. 특히 no-op 정본 선택, 전체 시트·열 dimension
복사 범위, 이미지 anchor·유효성 조건의 지원 범위를 먼저 명시한다. 기존 테스트가 통과한다는
이유로 현재 snapshot이 모든 Excel 속성을 검증한다고 해석하지 않는다.
