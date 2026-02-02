1. 인터넷에서 파이썬 설치.
2. 파일이 들은 폴더 오른쪽 클릭, '터미널에서 열기' 클릭
3. 아래 명령어 기입 후 진행.


pip install -r requirements.txt

4. 스크립트를 호출하여 검수 진행

호출 명령 양식, 원문 및 번역본 파일명, 용어집 파일명, 시작 언어 및 타겟언어 명칭 및 코드 잘 기입


python3 translation_checker_gemini_1.2.py \
  --source_file "(CX Center) SmartThings_2.0_Story_045_en.xlsx" \
  --target_file "(CX Center) SmartThings_2.0_Story_045_PTN.xlsx" \
  --range "C7:C28" \
  --src_lang "Korean" \
  --tgt_lang "English" \
  --src_code "한국어" \
  --tgt_code "영어_미국" \
  --glossary "glossary1.2.csv" \
  --sheet_langs_file "sheet_langs.json" \
  --max_concurrency 10 \
  --no_backtranslation


python3 translation_checker_gemini_1.2.py \
  --source_file "(CX Center) SmartThings_2.0_Story_051_en.xlsx" \
  --target_file "(CX Center) SmartThings_2.0_Story_051_PTN.xlsx" \
  --range "C7:C28" \
  --src_lang "English" \
  --tgt_lang "Korean" \
  --src_code "영어_미국" \
  --tgt_code "한국어" \
  --glossary "glossary1.2.csv" \
  --sheet_names "UK(영국),AU(호주),SG(싱가포르),FR(프랑스),BE(벨기에),CA(캐나다),DE(독일),IT(이탈리아),ES(스페인),NL(네덜란드),SE(스웨덴),AE(아랍에메리트),PT(포르투갈),BR(브라질),RU(러시아),TR(터키),PL(폴란드),VN(베트남),TH(태국),ID(인도네시아)" \
  --sheet_langs_file "sheet_langs.json" \
  --max_concurrency 10 \
  --no_backtranslation


python3 translation_checker_gemini_1.2.py \
  --source_file "(CX Center) SmartThings_2.0_Story_045_en.xlsx" \
  --target_file "(CX Center) SmartThings_2.0_Story_045_PTN.xlsx" \
  --range "C7:C28" \
  --src_lang "Korean" \
  --tgt_lang "English" \
  --src_code "한국어" \
  --tgt_code "영어_미국" \
  --glossary "glossary1.2.csv" \
  --sheet_names "UK(영국),AU(호주),SG(싱가포르)" \
  --sheet_langs_file "sheet_langs.json" \
  --max_concurrency 10 \
  --no_backtranslation



python3 translation_checker_gemini_1.2.py \
  --source_file "(CX Center) SmartThings_2.0_Story_045_en.xlsx" \
  --target_file "(CX Center) SmartThings_2.0_Story_045_PTN.xlsx" \
  --range "C7:C28" \
  --src_lang "English" \
  --tgt_lang "Korean" \
  --src_code "영어_미국" \
  --tgt_code "한국어" \
  --glossary "glossary1.2.csv" \
  --sheet_names "DE(독일),IT(이탈리아),ES(스페인),BR(브라질),PL(폴란드)" \
  --sheet_langs_file "sheet_langs.json" \
  --max_concurrency 10 \
  --no_backtranslation



#업데이트 내역
# 0.2 용어집 인식 수정, GPT 결제문제로 gpt 제거
# 0.3 비동기 처리 최적화
# 0.4 비동기 처리시 순서가 뒤죽박죽으로 기입되던 것 수정
# 0.6 용어집 내 규칙 내용도 프롬포트에 반영하도록 수정
# 0.7 검수 프롬포트 대문자 검수 강화, API Timeout 90초 설정
# 0.8 시트 이름(sheet_names) 인수를 통한 선택적 검수 기능 추가 및 버그 수정 (FINAL)
# 0.8A 세마포어(동시성 제한), 짧은 텍스트 화이트리스트, 용어집 사전 불일치 감지 추가
# 0.9 시트별 언어/코드 매핑(--sheet_langs / --sheet_langs_file) + 용어집 다언어 컬럼 지원 + 디버그
# 1.0 대소문자(문장형) 하드룰 강화 + 용어집 케이스 검수 + LLM 프롬프트에 케이스/고유명/기능명 평가·수정안 명시
#


  --sheet_names "UK(영국),AU(호주),SG(싱가포르),FR(프랑스),BE(벨기에),CA(캐나다),DE(독일),IT(이탈리아),ES(스페인),NL(네덜란드),SE(스웨덴),AE(아랍에메리트),PT(포르투갈),BR(브라질),RU(러시아),TR(터키),TW(대만),JA(일본),PL(폴란드),VN(베트남),TH(태국),ID(인도네시아)" \


python3 highlight_keywords.py glossary0.7.csv 프랑스어_캐나다 "(CX Center) Explore_2.0_Story_fr_CA.xlsx" "(CX Center) Explore_2.0_Story_fr_CA)_HL.xlsx"