import asyncio
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _app_pipeline as app_pipeline


def test_brazil_uses_portugal_glossary_targets_but_keeps_brazilian_language_rules():
    brazil = app_pipeline.DEFAULT_SHEET_LANGS["BR(브라질)"]

    assert brazil == {
        "lang": "Brazilian Portuguese",
        "code": "포르투갈_포르투갈",
    }


def test_app_ui_and_preview_defaults_match_brazil_glossary_override():
    app_root = ROOT.parents[1]
    index = (app_root / "src/translation_web_app/static/index.html").read_text(encoding="utf-8")
    main = (app_root / "src/translation_web_app/main.py").read_text(encoding="utf-8")

    assert '"BR(브라질)":  { "lang": "Brazilian Portuguese",  "code": "포르투갈_포르투갈" }' in index
    assert '"Brazilian Portuguese": "pt_PT"' in main


def test_brazil_default_resolves_portuguese_portugal_glossary_value(tmp_path):
    app_root = ROOT.parents[1]
    sys.path.insert(0, str(app_root / "src"))
    from translation_web_app.glossary_checks import GlossaryChecker
    from translation_web_app.prompt_builder import PromptBuilder

    glossary = tmp_path / "glossary.csv"
    glossary.write_text(
        ",,English_US,포르투갈_포르투갈,포르투갈_브라질\n"
        "key,Rule,en_US,pt_PT,pt_BR\n"
        "Lng,Rule,English_US,Portuguese,Portuguese_Latin\n"
        "Sample,,Sample,Portugal term,Brazil term\n",
        encoding="utf-8",
    )
    checker = GlossaryChecker(PromptBuilder())
    asyncio.run(checker.load_glossary_from_file(str(glossary), "English_US"))

    code = app_pipeline.DEFAULT_SHEET_LANGS["BR(브라질)"]["code"]
    context = checker._get_glossary_context_as_dict(code, source_text="Sample")

    assert context == {"Sample": "Portugal term"}
