"""Conservative AST ratchet for new Python workbook save bypasses.

This is a regression gate, not a sandbox: reflection/native/external writers
cannot be proven safe by static analysis. Imports alone never exempt a writer.
"""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

WATCHED = {"save", "save_workbook", "ExcelWriter", "to_excel"}


def scan(root: Path) -> list[dict]:
    findings = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # Watch attribute references too: `writer = wb.save; writer(path)` counts.
        for node in ast.walk(tree):
            if ((isinstance(node, ast.Attribute) and node.attr in WATCHED) or
                    (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in WATCHED)):
                expression = ast.dump(node, include_attributes=False)
                findings.append({"path": path.relative_to(root).as_posix(),
                    "expression_sha256": hashlib.sha256(expression.encode()).hexdigest(),
                    "expression": ast.unparse(node), "line": node.lineno})
    return findings


def violations(root: Path, baseline: dict) -> list[dict]:
    allowed = Counter((x["path"], x["expression_sha256"]) for x in baseline["sites"])
    unexpected = []
    for site in scan(root):
        key = site["path"], site["expression_sha256"]
        if allowed[key]:
            allowed[key] -= 1
        else:
            unexpected.append(site)
    return unexpected


def main():
    root = Path(__file__).resolve().parent
    baseline = json.loads((root.parent / "docs/workbook_writer_baseline.json").read_text())
    bad = violations(root, baseline)
    print(json.dumps({"status": "failed" if bad else "ok", "new_save_sites": bad}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(bad))


if __name__ == "__main__":
    main()
