from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path
from zipfile import ZipFile

import openpyxl
import pytest
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Font, PatternFill

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from workbook_mutation_guard import (WorkbookTextSafetyError, assert_serialized_text_safety,
    guard_external_workbook_saves,
    save_verified_atomic, save_workbook_atomic, _text_font_snapshot)


def fragile_book():
    book = openpyxl.Workbook()
    book.active.title = 'AE'
    for name in ('AE', 'Untouched'):
        sheet = book[name] if name in book.sheetnames else book.create_sheet(name)
        sheet['C17'].font = Font(name='Arial', sz=11, color='FF000000')
        sheet['C17'] = CellRichText('* ', TextBlock(InlineFont(color='FF0000FF', sz=11), 'SmartThings'),
            '\n', '* ', TextBlock(InlineFont(color='FFFF0000', sz=11), 'سطر أحمر'), '\r\n', ' tail ')
        sheet['C20'].fill = PatternFill('solid', fgColor='FFFFFF00')
        sheet['D1'] = CellRichText(' \n\t')
        sheet['D2'] = ' \n\t'
        sheet.row_dimensions[17].height = 100
    book['Untouched'].sheet_state = 'hidden'
    return book


def test_any_save_protects_untouched_sheets_and_retains_red_blue(tmp_path):
    book = fragile_book()
    expected = _text_font_snapshot(book)
    result = save_workbook_atomic(book, tmp_path / 'safe.xlsx')
    assert result['text_safety']['checks'] == {
        'whitespace': 'passed', 'linebreaks': 'passed', 'character_fonts': 'passed'}
    assert assert_serialized_text_safety(tmp_path / 'safe.xlsx')['status'] == 'passed'
    after = openpyxl.load_workbook(tmp_path / 'safe.xlsx', rich_text=True)
    assert _text_font_snapshot(after) == expected
    for name in book.sheetnames:
        assert copy.copy(after[name]['C20'].fill) == copy.copy(book[name]['C20'].fill)
        assert after[name].row_dimensions[17].height == 100
    assert after['Untouched'].sheet_state == 'hidden'
    first = (tmp_path / 'safe.xlsx').read_bytes()
    save_workbook_atomic(after, tmp_path / 'again.xlsx')
    second = openpyxl.load_workbook(tmp_path / 'again.xlsx', rich_text=True)
    assert _text_font_snapshot(second) == expected
    second.close(); after.close(); book.close()


@pytest.mark.parametrize('damage', ['space', 'linebreak', 'red_to_blue', 'font_size'])
def test_lossy_serializer_cannot_publish_even_with_a_pass_callback(tmp_path, damage):
    book = fragile_book()
    output = tmp_path / 'existing.xlsx'
    output.write_bytes(b'existing delivery must survive')
    original_save = book.save
    def bad_save(path):
        cell = book['AE']['C17']
        if damage in ('space', 'linebreak'):
            cell.value = str(cell.value).replace(' ' if damage == 'space' else '\n', '', 1)
        else:
            red = next(part for part in cell.value if isinstance(part, TextBlock)
                       and part.font.color is not None and part.font.color.type == 'rgb'
                       and part.font.color.rgb.endswith('FF0000'))
            if damage == 'red_to_blue':
                red.font = copy.copy(red.font); red.font.color = 'FF0000FF'
            else:
                red.font = copy.copy(red.font); red.font.sz = 12
        original_save(path)
    book.save = bad_save
    with pytest.raises(WorkbookTextSafetyError, match='mismatch'):
        save_verified_atomic(book, output, lambda path: {'status': 'passed'}, overwrite=True)
    assert output.read_bytes() == b'existing delivery must survive'
    assert not list(tmp_path.glob('.*.verify-*'))
    book.close()


def test_opaque_custom_writer_cannot_bypass_inspection(tmp_path):
    class Opaque:
        def save(self, path):
            pytest.fail('Opaque writer must not be called')
    with pytest.raises(WorkbookTextSafetyError, match='inspectable'):
        save_verified_atomic(Opaque(), tmp_path / 'bad.xlsx', lambda p: {'status': 'passed'})
    assert not (tmp_path / 'bad.xlsx').exists()


def test_raw_legacy_output_is_rejected_without_repair_at_delivery(tmp_path):
    book = fragile_book()
    path = tmp_path / 'unguarded.xlsx'
    book.save(path)  # Intentionally simulate an external, unguarded serializer.
    with pytest.raises(WorkbookTextSafetyError):
        assert_serialized_text_safety(path)
    book.close()


def test_verifier_cannot_modify_the_candidate_after_checking_it(tmp_path):
    book = fragile_book()
    def tamper(path):
        path.write_bytes(b'corrupt after verification')
        return {'status': 'passed'}
    with pytest.raises(WorkbookTextSafetyError, match='changed during'):
        save_verified_atomic(book, tmp_path / 'bad.xlsx', tamper)
    assert not (tmp_path / 'bad.xlsx').exists()
    book.close()


def test_failed_additional_verifier_cannot_save(tmp_path):
    book = fragile_book()
    with pytest.raises(WorkbookTextSafetyError, match='did not pass'):
        save_verified_atomic(book, tmp_path / 'bad.xlsx', lambda p: {'status': 'failed'})
    assert not (tmp_path / 'bad.xlsx').exists()
    book.close()


def test_package_has_no_direct_workbook_save_bypasses():
    violations = []
    for path in (ROOT / 'scripts').rglob('*.py'):
        if path.name == 'workbook_mutation_guard.py':
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'save':
                violations.append(f'{path.relative_to(ROOT)}:{node.lineno}')
    assert violations == [], 'Use the mandatory shared writer: ' + ', '.join(violations)


def test_external_app_saves_are_guarded_and_hook_is_restored(tmp_path):
    original = openpyxl.Workbook.save
    book = fragile_book()
    expected = _text_font_snapshot(book)
    with guard_external_workbook_saves():
        with guard_external_workbook_saves():
            book.save(tmp_path / 'app.xlsx')
        assert openpyxl.Workbook.save is not original
    assert openpyxl.Workbook.save is original
    reopened = openpyxl.load_workbook(tmp_path / 'app.xlsx', rich_text=True)
    assert _text_font_snapshot(reopened) == expected
    assert_serialized_text_safety(tmp_path / 'app.xlsx')
    reopened.close(); book.close()


def test_external_hook_is_restored_after_failure(tmp_path):
    original = openpyxl.Workbook.save
    book = fragile_book()
    with pytest.raises(RuntimeError):
        with guard_external_workbook_saves():
            raise RuntimeError('app failure')
    assert openpyxl.Workbook.save is original
    book.close()


def test_external_hook_does_not_capture_unrelated_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def unrelated_writer():
        book = fragile_book()
        try:
            book.save(tmp_path / 'unrelated.xlsx')
        finally:
            book.close()
    with guard_external_workbook_saves():
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(unrelated_writer).result()
    # An independent thread did not inherit the adapter's context. No global
    # application policy or unrelated workbook was silently changed.
    with pytest.raises(WorkbookTextSafetyError):
        assert_serialized_text_safety(tmp_path / 'unrelated.xlsx')


def test_even_the_whitespace_fixer_cannot_silently_recolour_text(tmp_path, monkeypatch):
    import workbook_incremental_highlight
    book = fragile_book()
    monkeypatch.setattr(workbook_incremental_highlight, '_fold_bare_whitespace_runs',
        lambda value: CellRichText(TextBlock(InlineFont(color='FF0000FF'), str(value))))
    with pytest.raises(WorkbookTextSafetyError, match='visible character'):
        save_workbook_atomic(book, tmp_path / 'bad.xlsx')
    assert not (tmp_path / 'bad.xlsx').exists()
    book.close()
