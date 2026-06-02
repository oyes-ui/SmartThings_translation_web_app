import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from translation_web_app.checker_service import TranslationChecker
from translation_web_app.routers.glossaries import create_glossaries_router
from translation_web_app.services.glossary_store import GlossaryStore


def write_glossary_csv(path: Path) -> None:
    path.write_text(
        "\ufeffkey,Rule,한국어,영어_미국,독어_독일\n"
        ",,ko_KR,en_US,de_DE\n"
        "Lng,,Korean,English_US,German\n"
        "SmartThings,대괄호 제외,스마트싱스,SmartThings,SmartThings\n"
        "AI,비활성화,AI,AI,KI\n",
        encoding="utf-8",
    )


def test_glossary_store_import_export_and_checker_load(tmp_path: Path):
    csv_path = tmp_path / "seed.csv"
    db_path = tmp_path / "glossary.db"
    export_path = tmp_path / "export.csv"
    write_glossary_csv(csv_path)

    store = GlossaryStore(db_path=db_path, seed_csv=csv_path)
    result = store.import_csv(csv_path, mode="replace")
    assert result["created"] == 2
    assert store.status()["term_count"] == 2
    assert len(store.list_locales()) == 3

    exported = store.export_csv(export_path)
    checker = TranslationChecker()
    message = asyncio.run(checker.load_glossary_from_file(str(exported), "Korean"))

    assert "용어집 로드 성공" in message
    assert checker.glossary["SmartThings"]["targets"]["독어_독일"] == "SmartThings"
    assert checker.glossary["SmartThings"]["rule"] == "대괄호 제외"
    assert checker.glossary["AI"]["rule"] == "비활성화"


def test_glossary_store_seed_imports_only_when_empty(tmp_path: Path):
    csv_path = tmp_path / "latest_glossary.csv"
    db_path = tmp_path / "glossary.db"
    write_glossary_csv(csv_path)

    store = GlossaryStore(db_path=db_path, seed_csv=csv_path)
    assert store.ensure_seeded() is True
    assert store.ensure_seeded() is False
    assert store.status()["term_count"] == 2


def test_glossary_store_merge_and_replace(tmp_path: Path):
    csv_path = tmp_path / "seed.csv"
    db_path = tmp_path / "glossary.db"
    write_glossary_csv(csv_path)
    store = GlossaryStore(db_path=db_path, seed_csv=csv_path)
    store.import_csv(csv_path, mode="replace")

    merge_path = tmp_path / "merge.csv"
    merge_path.write_text(
        "\ufeffkey,Rule,한국어,영어_미국\n"
        ",,ko_KR,en_US\n"
        "Lng,,Korean,English_US\n"
        "SmartThings,동일 의미 모두 적용,스마트싱스,SmartThings\n"
        "Matter,,매터,Matter\n",
        encoding="utf-8",
    )
    result = store.import_csv(merge_path, mode="merge")
    assert result["created"] == 1
    assert result["updated"] == 1
    assert store.status()["term_count"] == 3

    store.import_csv(merge_path, mode="replace")
    assert store.status()["term_count"] == 2


def test_glossary_router_crud_and_import(tmp_path: Path):
    seed_path = tmp_path / "missing.csv"
    db_path = tmp_path / "glossary.db"

    def factory():
        return GlossaryStore(db_path=db_path, seed_csv=seed_path)

    app = FastAPI()
    app.include_router(create_glossaries_router(factory))
    client = TestClient(app)

    csv_path = tmp_path / "upload.csv"
    write_glossary_csv(csv_path)
    with csv_path.open("rb") as f:
        response = client.post("/api/glossary/import?mode=replace", files={"file": ("glossary.csv", f, "text/csv")})
    assert response.status_code == 200
    assert response.json()["created"] == 2

    terms = client.get("/api/glossary/terms").json()["terms"]
    first = terms[0]
    first["rule_text"] = "대괄호 제외; 동일 의미 모두 적용"
    response = client.put(f"/api/glossary/terms/{first['id']}", json=first)
    assert response.status_code == 200

    response = client.post(
        "/api/glossary/terms",
        json={"source_key": "Matter", "rule_text": "", "translations": {}},
    )
    assert response.status_code == 200
    new_id = response.json()["id"]

    response = client.delete(f"/api/glossary/terms/{new_id}")
    assert response.status_code == 200

    response = client.get("/api/glossary/export.csv")
    assert response.status_code == 200
    assert "SmartThings" in response.text
