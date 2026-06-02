"""SQLite-backed glossary store with CSV import/export compatibility."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from translation_web_app.paths import GLOSSARY_DB_PATH, LATEST_GLOSSARY_CSV, UPLOAD_DIR


RULE_HEADER_MARKERS = ("설명", "규칙", "rule", "비고", "note", "remark", "desc")


@dataclass(frozen=True)
class GlossaryResolution:
    path: str | None
    source: str
    message: str


class GlossaryStore:
    """Manage the built-in glossary while preserving the existing 3-row CSV shape."""

    def __init__(self, db_path: Path = GLOSSARY_DB_PATH, seed_csv: Path = LATEST_GLOSSARY_CSV):
        self.db_path = Path(db_path)
        self.seed_csv = Path(seed_csv)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS glossary_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS glossary_locales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    header_display TEXT NOT NULL,
                    header_code TEXT NOT NULL DEFAULT '',
                    header_alias TEXT NOT NULL DEFAULT '',
                    column_order INTEGER NOT NULL,
                    UNIQUE(header_display, header_code, header_alias)
                );

                CREATE TABLE IF NOT EXISTS glossary_terms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_key TEXT NOT NULL UNIQUE,
                    rule_text TEXT NOT NULL DEFAULT '',
                    sort_order INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS glossary_translations (
                    term_id INTEGER NOT NULL REFERENCES glossary_terms(id) ON DELETE CASCADE,
                    locale_id INTEGER NOT NULL REFERENCES glossary_locales(id) ON DELETE CASCADE,
                    value TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(term_id, locale_id)
                );
                """
            )

    def ensure_seeded(self) -> bool:
        """Import the seed CSV once when the DB has no terms."""
        if self.term_count() > 0 or not self.seed_csv.exists():
            return False
        self.import_csv(self.seed_csv, mode="replace")
        return True

    def term_count(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM glossary_terms").fetchone()[0])

    def status(self) -> dict[str, Any]:
        self.ensure_seeded()
        with self.connect() as conn:
            locale_count = int(conn.execute("SELECT COUNT(*) FROM glossary_locales").fetchone()[0])
            term_count = int(conn.execute("SELECT COUNT(*) FROM glossary_terms").fetchone()[0])
        return {
            "db_path": str(self.db_path),
            "seed_csv": str(self.seed_csv),
            "seed_exists": self.seed_csv.exists(),
            "term_count": term_count,
            "locale_count": locale_count,
        }

    def import_csv(self, csv_path: Path, mode: str = "merge") -> dict[str, int | str]:
        if mode not in {"merge", "replace"}:
            raise ValueError("mode must be 'merge' or 'replace'")

        rows = self._read_csv_rows(csv_path)
        if len(rows) < 3:
            raise ValueError("Glossary CSV must include 3 header rows.")

        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        header_rows = rows[:3]
        source_col = 0
        rule_col = self._detect_rule_col(header_rows)
        data_rows = rows[3:]

        locale_cols = [
            idx for idx in range(width)
            if idx not in {source_col, rule_col}
            and any(header_rows[r][idx].strip() for r in range(3))
        ]

        with self.connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            if mode == "replace":
                conn.execute("DELETE FROM glossary_translations")
                conn.execute("DELETE FROM glossary_terms")
                conn.execute("DELETE FROM glossary_locales")
                conn.execute("DELETE FROM glossary_meta")

            for r in range(3):
                conn.execute(
                    "INSERT OR REPLACE INTO glossary_meta(key, value) VALUES(?, ?)",
                    (f"source_header_{r}", header_rows[r][source_col].strip()),
                )
                if rule_col != -1:
                    conn.execute(
                        "INSERT OR REPLACE INTO glossary_meta(key, value) VALUES(?, ?)",
                        (f"rule_header_{r}", header_rows[r][rule_col].strip()),
                    )

            locale_id_by_col: dict[int, int] = {}
            for order, col in enumerate(locale_cols):
                display = header_rows[0][col].strip()
                code = header_rows[1][col].strip()
                alias = header_rows[2][col].strip()
                locale_id_by_col[col] = self._upsert_locale(conn, display, code, alias, order)

            imported = 0
            updated = 0
            for order, row in enumerate(data_rows):
                source_key = row[source_col].strip() if source_col < len(row) else ""
                if not source_key or source_key.lower() == "lng":
                    continue
                rule_text = row[rule_col].strip() if rule_col != -1 and rule_col < len(row) else ""
                term_id, created = self._upsert_term(conn, source_key, rule_text, order)
                imported += int(created)
                updated += int(not created)
                for col, locale_id in locale_id_by_col.items():
                    value = row[col].strip() if col < len(row) else ""
                    conn.execute(
                        """
                        INSERT INTO glossary_translations(term_id, locale_id, value)
                        VALUES(?, ?, ?)
                        ON CONFLICT(term_id, locale_id) DO UPDATE SET value = excluded.value
                        """,
                        (term_id, locale_id, value),
                    )

        return {"mode": mode, "created": imported, "updated": updated, "locales": len(locale_cols)}

    def list_locales(self) -> list[dict[str, Any]]:
        self.ensure_seeded()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, header_display, header_code, header_alias, column_order
                FROM glossary_locales
                ORDER BY column_order, id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_terms(self, search: str = "", limit: int = 200, offset: int = 0) -> dict[str, Any]:
        self.ensure_seeded()
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        pattern = f"%{search.strip()}%"

        with self.connect() as conn:
            if search.strip():
                where = "WHERE source_key LIKE ? OR rule_text LIKE ?"
                count = int(conn.execute(f"SELECT COUNT(*) FROM glossary_terms {where}", (pattern, pattern)).fetchone()[0])
                terms = conn.execute(
                    f"""
                    SELECT id, source_key, rule_text, sort_order
                    FROM glossary_terms {where}
                    ORDER BY sort_order, id
                    LIMIT ? OFFSET ?
                    """,
                    (pattern, pattern, limit, offset),
                ).fetchall()
            else:
                count = int(conn.execute("SELECT COUNT(*) FROM glossary_terms").fetchone()[0])
                terms = conn.execute(
                    """
                    SELECT id, source_key, rule_text, sort_order
                    FROM glossary_terms
                    ORDER BY sort_order, id
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()

            term_ids = [int(row["id"]) for row in terms]
            translations: dict[int, dict[str, str]] = {term_id: {} for term_id in term_ids}
            if term_ids:
                placeholders = ",".join("?" for _ in term_ids)
                for row in conn.execute(
                    f"""
                    SELECT gt.term_id, gl.id AS locale_id, gt.value
                    FROM glossary_translations gt
                    JOIN glossary_locales gl ON gl.id = gt.locale_id
                    WHERE gt.term_id IN ({placeholders})
                    """,
                    term_ids,
                ).fetchall():
                    translations[int(row["term_id"])][str(row["locale_id"])] = row["value"]

        return {
            "total": count,
            "limit": limit,
            "offset": offset,
            "terms": [
                {
                    "id": int(row["id"]),
                    "source_key": row["source_key"],
                    "rule_text": row["rule_text"],
                    "translations": translations.get(int(row["id"]), {}),
                }
                for row in terms
            ],
        }

    def create_term(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_key = str(payload.get("source_key") or "").strip()
        if not source_key:
            raise ValueError("source_key is required")
        rule_text = str(payload.get("rule_text") or "").strip()
        translations = payload.get("translations") or {}
        with self.connect() as conn:
            next_order = int(conn.execute("SELECT COALESCE(MAX(sort_order), -1) + 1 FROM glossary_terms").fetchone()[0])
            term_id, _ = self._upsert_term(conn, source_key, rule_text, next_order)
            self._replace_translations(conn, term_id, translations)
        return {"id": term_id, "source_key": source_key, "rule_text": rule_text, "translations": translations}

    def update_term(self, term_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        source_key = str(payload.get("source_key") or "").strip()
        if not source_key:
            raise ValueError("source_key is required")
        rule_text = str(payload.get("rule_text") or "").strip()
        translations = payload.get("translations") or {}
        with self.connect() as conn:
            existing = conn.execute("SELECT id FROM glossary_terms WHERE id = ?", (term_id,)).fetchone()
            if not existing:
                raise KeyError("term not found")
            conn.execute(
                "UPDATE glossary_terms SET source_key = ?, rule_text = ? WHERE id = ?",
                (source_key, rule_text, term_id),
            )
            self._replace_translations(conn, term_id, translations)
        return {"id": term_id, "source_key": source_key, "rule_text": rule_text, "translations": translations}

    def delete_term(self, term_id: int) -> None:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM glossary_terms WHERE id = ?", (term_id,))
            if cur.rowcount == 0:
                raise KeyError("term not found")

    def export_csv(self, output_path: Path) -> Path:
        self.ensure_seeded()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.connect() as conn:
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM glossary_meta")}
            locales = conn.execute(
                """
                SELECT id, header_display, header_code, header_alias
                FROM glossary_locales
                ORDER BY column_order, id
                """
            ).fetchall()
            terms = conn.execute(
                "SELECT id, source_key, rule_text FROM glossary_terms ORDER BY sort_order, id"
            ).fetchall()
            values = {
                (int(row["term_id"]), int(row["locale_id"])): row["value"]
                for row in conn.execute("SELECT term_id, locale_id, value FROM glossary_translations")
            }

        header_rows = [
            [meta.get("source_header_0", "key"), meta.get("rule_header_0", "Rule")],
            [meta.get("source_header_1", ""), meta.get("rule_header_1", "")],
            [meta.get("source_header_2", "Lng"), meta.get("rule_header_2", "")],
        ]
        for locale in locales:
            header_rows[0].append(locale["header_display"])
            header_rows[1].append(locale["header_code"])
            header_rows[2].append(locale["header_alias"])

        with output_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(header_rows)
            for term in terms:
                row = [term["source_key"], term["rule_text"]]
                row.extend(values.get((int(term["id"]), int(locale["id"])), "") for locale in locales)
                writer.writerow(row)

        return output_path

    def export_temp_csv(self) -> Path | None:
        if self.term_count() == 0:
            self.ensure_seeded()
        if self.term_count() == 0:
            return None
        output = UPLOAD_DIR / f"glossary_builtin_{uuid4()}.csv"
        return self.export_csv(output)

    @staticmethod
    def _read_csv_rows(csv_path: Path) -> list[list[str]]:
        with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
            return [[str(cell) for cell in row] for row in csv.reader(f)]

    @staticmethod
    def _detect_rule_col(header_rows: list[list[str]]) -> int:
        width = max(len(row) for row in header_rows)
        for col in range(width):
            values = " ".join(row[col].strip().lower() for row in header_rows if col < len(row))
            if any(marker in values for marker in RULE_HEADER_MARKERS):
                return col
        return 1 if width > 1 else -1

    @staticmethod
    def _upsert_locale(conn: sqlite3.Connection, display: str, code: str, alias: str, order: int) -> int:
        existing = conn.execute(
            """
            SELECT id FROM glossary_locales
            WHERE header_display = ? AND header_code = ? AND header_alias = ?
            """,
            (display, code, alias),
        ).fetchone()
        if existing:
            conn.execute("UPDATE glossary_locales SET column_order = ? WHERE id = ?", (order, existing["id"]))
            return int(existing["id"])

        cur = conn.execute(
            """
            INSERT INTO glossary_locales(header_display, header_code, header_alias, column_order)
            VALUES(?, ?, ?, ?)
            """,
            (display, code, alias, order),
        )
        return int(cur.lastrowid)

    @staticmethod
    def _upsert_term(conn: sqlite3.Connection, source_key: str, rule_text: str, order: int) -> tuple[int, bool]:
        existing = conn.execute("SELECT id FROM glossary_terms WHERE source_key = ?", (source_key,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE glossary_terms SET rule_text = ? WHERE id = ?",
                (rule_text, existing["id"]),
            )
            return int(existing["id"]), False

        cur = conn.execute(
            "INSERT INTO glossary_terms(source_key, rule_text, sort_order) VALUES(?, ?, ?)",
            (source_key, rule_text, order),
        )
        return int(cur.lastrowid), True

    @staticmethod
    def _replace_translations(conn: sqlite3.Connection, term_id: int, translations: dict[str, Any]) -> None:
        locale_ids = [int(key) for key in translations.keys() if str(key).isdigit()]
        if locale_ids:
            placeholders = ",".join("?" for _ in locale_ids)
            conn.execute(
                f"DELETE FROM glossary_translations WHERE term_id = ? AND locale_id IN ({placeholders})",
                [term_id, *locale_ids],
            )
        for locale_id, value in translations.items():
            if not str(locale_id).isdigit():
                continue
            conn.execute(
                """
                INSERT INTO glossary_translations(term_id, locale_id, value)
                VALUES(?, ?, ?)
                ON CONFLICT(term_id, locale_id) DO UPDATE SET value = excluded.value
                """,
                (term_id, int(locale_id), str(value or "").strip()),
            )


def resolve_glossary_file(uploaded_glossary_id: str | None) -> GlossaryResolution:
    """Return the glossary CSV path following uploaded CSV > built-in DB > none."""
    if uploaded_glossary_id:
        uploaded_path = UPLOAD_DIR / uploaded_glossary_id
        return GlossaryResolution(
            path=str(uploaded_path),
            source="uploaded",
            message=f"Using uploaded glossary CSV: {uploaded_glossary_id}",
        )

    store = GlossaryStore()
    exported = store.export_temp_csv()
    if exported:
        return GlossaryResolution(
            path=str(exported),
            source="builtin",
            message="Using built-in glossary DB.",
        )

    return GlossaryResolution(path=None, source="none", message="No glossary selected.")
