# SmartThings Story Contents: 역할별 표기·대소문자 규칙

이 문서는 2026-08-07 기준 SmartThings Story Contents 워크북의 실제 검수 기준을 Excel live 작업에 명시적으로 전달하기 위한 정적 규칙이다. 특정 셀 주소가 아니라 **콘텐츠 역할**을 우선한다.

## 1. 역할 판별

수정 또는 검수 전에 대상 텍스트를 아래 중 하나로 분류한다.

| 역할 | 판별 기준 | 기본 처리 |
| --- | --- | --- |
| Story/section title | story 제목, section heading, 카드 제목; 일반적으로 C7/C10/C15/C20/C25 계열 | 제목 문구 규칙 적용 |
| Body/description | 제목을 설명·보완하는 한 문장 이상 본문 | 해당 언어의 일반 문장 규칙 적용 |
| Button/CTA | 누르면 행동이 발생하는 짧은 UI 문구 | 간결한 명령/동사형, 언어별 case 적용 |
| UI navigation path | 메뉴·탭·설정 경로를 순서대로 인용 | 언어별 따옴표/기호·glossary 규칙 적용 |
| Legal/disclaimer | 법적 고지, 조건, 권리 고지 | 원문 의미·문장 부호·시장별 legal 표현 보존 |
| Glossary/product term | 제품명, 기능명, 등록 용어 | glossary 표기 우선 |

주소가 title 행처럼 보여도, 실제 내용이 body/list/legal이면 역할을 재판정한다. 반대로 병합·재배치된 셀은 주소가 달라도 title일 수 있다.

## 2. 제목·헤딩의 끝 문장부호

- Story/section title과 heading은 문장처럼 보이더라도 마지막 마침표 `.` 또는 `。`를 쓰지 않는다.
- 의문형 제목의 `?`/`？`는 유지한다. 예: `¿Luces? Encendidas. ¿Ánimo? Por las nubes`.
- 느낌표, 생략 부호, URL, 파일 확장자, 제품 모델명, 약어, 숫자/버전은 맥락을 확인한 뒤 변경한다. 단순 규칙으로 삭제하지 않는다.
- Body, legal/disclaimer, list의 완전 문장에는 이 규칙을 적용하지 않는다. 해당 언어 typography 규칙을 따른다.
- title에 glossary term이 포함되어도 glossary의 대소문자·공백을 그대로 유지한다.

## 3. Casing 우선순위

1. Workbook의 승인된 glossary 표기(대소문자·공백·시장 변형 포함)
2. 대상 언어 규칙 (`canonical-rules/languages/`)
3. 콘텐츠 역할별 규칙
4. 일반 typography/BX 선호

따라서 제목·버튼이라도 glossary의 `SmartThings Energy`, `Galaxy Watch` 같은 표기를 title case나 sentence case에 맞춘다는 이유로 바꾸지 않는다.

## 4. 언어별 대표 case 주의점

- **es_CO / Spanish**: 영어식 단어별 Title Case를 만들지 않는다. 자연스러운 스페인어 sentence case를 사용하되 glossary 표기는 보존한다.
- **Swedish / Vietnamese / French / Russian**: 영어식 Title Case 또는 불필요한 대문자를 피한다. 각각의 언어 규칙 파일을 우선한다.
- **German**: 명사의 정상 대문자화와 합성어 구조를 유지한다. title을 이유로 정상 독일어 대소문자를 무너뜨리지 않는다.
- **Simplified/Traditional Chinese**: 전각 문장부호를 사용한다. 중국어용 인용 부호 규칙은 각 언어 규칙을 따른다.
- **English**: 시장 변형(US/UK/AU/SG)과 disclaimer의 마침표 위치를 언어 규칙 파일에서 확인한다.

## 5. Preview / audit 출력

case 또는 문장부호를 제안할 때는 다음 정보를 함께 보여 준다.

```json
{
  "sheet": "CO(콜롬비아)",
  "cell": "C15",
  "content_role": "section_title",
  "current": "El horno se configura automáticamente según tu receta.",
  "proposed": "El horno se configura automáticamente según tu receta",
  "rule": "Title/section heading: terminal period is removed; es_CO uses natural sentence case",
  "glossary_override": false,
  "status": "preview"
}
```

역할, glossary 우선 여부, locale 중 하나라도 불명확하면 `needs_review`로 표시하고 자동 수정하지 않는다.
