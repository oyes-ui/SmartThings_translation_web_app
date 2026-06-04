# First-run Setup Workflow

이 skill 은 standalone 이 아니라 **SmartThings Translation Web App repo 를 참조하는 app-aware 에이전트**다. 번역 skill 실행 세션은 skill 패키지 폴더에서 시작하고, app repo 는 `--app-root` 또는 저장 config 로 연결한다.

## 트리거

- "SmartThings 번역 에이전트 시작해줘"
- "셋업 / setup / bootstrap 해줘"
- RAG/규칙/full 기능이 필요한 요청인데 아직 app_root 가 확인되지 않은 경우

## 절차

```
1. bootstrap 실행
   python scripts/bootstrap.py --app-root <app-repo-path> --json

2. app_root 해석 (bootstrap 내부 우선순위)
   ① --app-root 인자
   ② 환경변수 SMARTTHINGS_APP_ROOT
   ③ 저장된 config (~/.smartthings_translation_agent/config.json)
   ④ 현재 cwd 및 상위 탐색 (보조 fallback)
   ⑤ skill 파일 위치 상위 탐색 (repo 안 설치용 fallback)

3. 결과 분기
   ├─ app_root 미발견 (app_root_source == "not-found")
   │    → 사용자에게 app 폴더 경로를 한 번 요청
   │    → 받은 경로로:  python scripts/bootstrap.py --app-root <경로> --save
   │    → 이후 호출은 저장된 config 를 자동 사용
   │
   └─ app_root 발견
        → checks 와 level 을 사용자에게 요약 표시
        → 부족한 것 안내 (아래 "상태 점검" 참조)
```

## 상태 점검 (bootstrap checks → 안내)

| 항목 | 없을 때 안내 |
|------|--------------|
| `venv_python` | semantic/full 기능은 의존성 설치 필요. `pip install -r requirements-rag.txt` 또는 app `requirements.txt` 설치를 **승인 후** 안내 |
| `rag_store_db` | offline/semantic RAG 제한. RAG DB 전달받거나 `rag_db_builder --build-all`(크레딧 소모, **승인 필요**) |
| `chroma_dir` | semantic 유사검색 불가. offline(exact/keyword)은 가능 |
| `api_key_present` | semantic 불가. `.env` 에 본인 `GEMINI_API_KEY` 입력 안내. offline 은 키 없이 가능 |
| `glossary_csv`/`glossary_db` | 용어집 기반 검증 제한. CSV 전달/업로드 안내 |

## 동작 레벨 (bootstrap 이 판정)

| Level | 조건 | 가능한 것 |
|-------|------|-----------|
| 1 Excel-only | openpyxl 만 | 워크북 분석/수정, section coherence |
| 2 Offline RAG | app_root + `rag_store.db` | + exact/keyword/메타 RAG (키 불필요) |
| 3 Semantic RAG | + venv + chroma + API key | + 임베딩 유사검색 |
| 4 Full pipeline | + glossary | + 용어집 검증/full app 연계 |

## 안전 원칙 (셋업 단계)

- 의존성 설치, RAG DB 빌드, app 실행 등 **크레딧/변경을 수반하는 작업은 항상 사용자 승인 후** 진행한다.
- API 키 값은 출력/로그에 남기지 않는다. bootstrap 은 키 **존재 여부(boolean)** 만 보고한다.
- 자동 clone 은 기본 흐름이 아니다. 로컬 app 폴더 지정/저장이 기본이다.
- app 유지보수/오케스트레이션 파일은 번역 skill 실행 세션의 기본 스코프가 아니다.

## 이후 사용 (재방문)

`config.json` 에 `app_root` 가 저장돼 있으면 bootstrap 없이도 각 스크립트가 이를 자동 사용한다(`--app-root` 미지정 시). 사용자는 곧장 작업 요청만 하면 된다:

```
"이 Excel에서 JA 시트 title이 description 맥락을 잘 반영했는지 봐줘"
  → workbook_inspect.py --sections (저장된 app_root 자동 사용)
  → 필요 시 offline RAG 조회
  → 제안만, 승인 후 workbook_apply_edits.py
```
