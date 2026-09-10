"""Validated DB -> default CSV refresh; existing job snapshots remain unchanged."""
import csv
import os
from pathlib import Path
from workbook_contract import atomic_json, file_sha256
from workbook_run import exclusive_lock


def sync_default(store, destination: Path) -> dict:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = destination.with_suffix(".sync.json")
    with exclusive_lock(destination.with_suffix(".sync.lock")):
        temporary = destination.with_suffix(".csv.tmp")
        check = destination.with_suffix(".check.tmp")
        atomic_json(marker, {"status": "syncing"})
        try:
            store.export_csv(temporary)
            with temporary.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            if len(rows) < 3 or len(rows[0]) < 3 or any(len(r) != len(rows[0]) for r in rows):
                raise ValueError("Default glossary must have three rectangular header rows and locale columns")
            # A second DB read detects mutations during export; compare the serialized canonical view.
            store.export_csv(check)
            if temporary.read_bytes() != check.read_bytes():
                raise ValueError("Glossary DB changed during export; run sync-default again")
            sha = file_sha256(temporary)
            os.replace(temporary, destination)
            result = {"status": "ok", "csv": str(destination), "sha256": sha, "term_count": len(rows) - 3}
            atomic_json(marker, result)
            return result
        except Exception as error:
            atomic_json(marker, {"status": "failed", "error": str(error), "next_action": "sync-default"})
            raise
        finally:
            temporary.unlink(missing_ok=True)
            check.unlink(missing_ok=True)
