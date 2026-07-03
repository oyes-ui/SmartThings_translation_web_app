import sys
from pathlib import Path
import openpyxl
import pytest

# Add the script folder to sys.path
agent_scripts_dir = Path(__file__).parent.parent / "agent-packages" / "smartthings-translation-agent" / "scripts"
sys.path.append(str(agent_scripts_dir))

import review_summary


def test_classify_judgment():
    assert review_summary._classify_judgment("수용 비추천") == "reject"
    assert review_summary._classify_judgment("미반영") == "reject"
    assert review_summary._classify_judgment("부분 수용") == "partial"
    assert review_summary._classify_judgment("부분 반영") == "partial"
    assert review_summary._classify_judgment("추가 적용 없음") == "no_apply"
    assert review_summary._classify_judgment("유지") == "no_apply"
    assert review_summary._classify_judgment("전체 수용") == "accept"
    assert review_summary._classify_judgment("수용 가능") == "accept"
    assert review_summary._classify_judgment("아무 말") == "conditional_or_other"


def test_parse_ai_review(tmp_path: Path):
    content = """
[시트] DE(독일) | [셀] C10
최종 평가: 수정 필요
[수정안 제안]:
Neues Wort

[상세 설명]
어쩌구

[시트] FR(프랑스) | [셀] C12
최종 평가: 양호
[수정안 제안]:

[상세 설명]
어쩌구

[시트] IT(이탈리아) | [셀] C15
최종 평가: 우수
[수정안 제안]:
Nuovo

====================
"""
    ai_file = tmp_path / "ai_review.txt"
    ai_file.write_text(content, encoding="utf-8")

    result, suggested = review_summary.parse_ai_review(ai_file)
    assert result is not None
    assert result["ai_blocks"] == 3
    assert result["ai_grade_counts"] == {
        "Needs Revision": 1,
        "Good": 1,
        "Excellent": 1
    }
    assert result["ai_suggested_fix_count"] == 2
    assert suggested == {("DE(독일)", 10), ("IT(이탈리아)", 15)}


def test_parse_report_md(tmp_path: Path):
    content = """
# 검수 리포트
## 10. 재감수 대조 기록
| 셀 좌표 | 원본 | 번역사안 | 최종 판단 |
|---|---|---|---|
| `DE C10` | Old | New | 수용 비추천 |
| `FR C12` | Hello | Hi | 전체 수용 |
| `IT C15` | Ciao | Salve | 부분 수용 |
| `ES C18` | Hola | Buenos | 유지 |
| `GB C20` | Color | Colour | 기타 |

## 11. 최종 Summary
요약 정보
"""
    report_file = tmp_path / "report.md"
    report_file.write_text(content, encoding="utf-8")

    result = review_summary.parse_report_md(report_file)
    assert result is not None
    assert result["report_rows"] == 5
    assert result["report_classification_counts"] == {
        "reject": 1,
        "accept": 1,
        "partial": 1,
        "no_apply": 1,
        "conditional_or_other": 1
    }


def test_summarize_review_workbook(tmp_path: Path):
    wb_path = tmp_path / "review_wb.xlsx"
    wb = openpyxl.Workbook()
    
    # Setup source sheets
    ws_kr = wb.active
    ws_kr.title = "KR(한국)"
    
    # Setup target sheet
    ws_de = wb.create_sheet("DE(독일)")
    
    # Populate KR sheet to be excluded
    ws_kr["C7"] = "한국어 번역 대상"
    
    # Populate target sheet
    # 1. Changed cell
    ws_de["C7"] = "Original DE"
    ws_de["F7"] = "Revised DE"
    
    # 2. No change cell
    ws_de["C8"] = "Same text"
    ws_de["F8"] = "Same text"
    
    # 3. Commented cell
    ws_de["C9"] = "Another text"
    ws_de["H9"] = "Comment here"
    
    # 4. Empty target cells (none)
    
    wb.save(wb_path)
    wb.close()
    
    result, changed = review_summary.summarize_review_workbook(
        wb_path,
        count_path=None,
        source_sheets={"KR(한국)", "US(미국)"}
    )
    
    assert result["sheets_counted"] == 1
    assert result["total_target_cells"] == 3
    assert result["reviewer_changed_cells"] == 1
    assert result["reviewer_touched_cells"] == 3
    assert result["comment_cells"] == 1
    assert changed == {("DE(독일)", 7)}
