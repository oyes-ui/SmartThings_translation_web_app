import openpyxl
import csv
import re
import sys
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont


# 1️⃣ CSV에서 단어 리스트 불러오기
def load_keywords_from_csv(filename, key_column):
    keywords = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            keyword = row.get(key_column)
            if keyword:
                keywords.append(keyword.strip())
    return sorted(keywords, key=len, reverse=True)


# 2️⃣ 워크북 내 단어 하이라이트 처리
def highlight_keywords_in_workbook(input_file, output_file, keywords):
    from openpyxl import load_workbook
    from openpyxl.cell.rich_text import CellRichText, TextBlock
    from openpyxl.cell.text import InlineFont

    wb = load_workbook(input_file)
    for ws in wb.worksheets:
        print(f"🔍 워크시트 처리 중: {ws.title}")
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    text = cell.value
                    # 키워드 일치 탐색
                    matches = list(re.finditer('|'.join(re.escape(k) for k in keywords), text, flags=re.IGNORECASE))
                    if not matches:
                        continue

                    parts = []
                    last_end = 0
                    for m in matches:
                        start, end = m.span()
                        if start > last_end:
                            parts.append((text[last_end:start], InlineFont(color="000000")))
                        parts.append((text[start:end], InlineFont(color="0000FF")))  # ✅ 파란색으로 단어만 표시
                        last_end = end
                    if last_end < len(text):
                        parts.append((text[last_end:], InlineFont(color="000000")))

                    rt = CellRichText()
                    for segment_text, segment_font in parts:
                        tb = TextBlock(text=segment_text, font=segment_font)
                        rt.append(tb)
                    cell.value = rt

    wb.save(output_file)
    print(f"✅ 완료! 결과 파일: {output_file}")


# 3️⃣ 실행부
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("\n⚙️ 사용법:")
        print("python3 highlight_keywords.py <CSV_FILE> <COLUMN_KEY> <INPUT_XLSX> <OUTPUT_XLSX>")
        print("\n예시:")
        print('python3 highlight_keywords.py glossary0.5.csv 아랍에미리트 "(CX Center) Explore_2.0_Story_ar_AE.xlsx" "(CX Center) Explore_2.0_Story_ar_AE_HL.xlsx"\n')
        sys.exit(1)

    CSV_FILENAME = sys.argv[1]
    CSV_KEY_COLUMN = sys.argv[2]
    INPUT_EXCEL_FILE = sys.argv[3]
    OUTPUT_EXCEL_FILE = sys.argv[4]

    print(f"📘 CSV 파일: {CSV_FILENAME}")
    print(f"🔑 번역 키: {CSV_KEY_COLUMN}")
    print(f"📂 입력 파일: {INPUT_EXCEL_FILE}")
    print(f"💾 출력 파일: {OUTPUT_EXCEL_FILE}")

    keywords = load_keywords_from_csv(CSV_FILENAME, CSV_KEY_COLUMN)
    highlight_keywords_in_workbook(INPUT_EXCEL_FILE, OUTPUT_EXCEL_FILE, keywords)