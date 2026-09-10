"""Versioned, immutable execution contracts. No workbook mutation lives here."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

AXES = ("values", "layout", "styles", "rich_text", "annotations")
ROLES = {"values": "values", "layout": "structure", "styles": "formatting",
         "rich_text": "formatting", "annotations": "formatting"}
SCHEMA_VERSION = 1


class ContractError(ValueError):
    code = "contract_invalid"


class InputDrift(ContractError):
    code = "input_drift"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha256(path: Path | str) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, canonical(payload) + "\n")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def inventory(path: Path | str) -> dict:
    path = Path(path)
    if path.is_symlink():
        raise ContractError("Authority symlinks are not supported")
    path = path.resolve(strict=True)
    if path.is_file():
        files = [{"path": ".", "sha256": file_sha256(path)}]
        kind = "file"
    elif path.is_dir():
        files = []
        for child in sorted(path.rglob("*")):
            if child.is_symlink():
                raise ContractError("Authority directory contains a symlink")
            if child.is_file():
                files.append({"path": child.relative_to(path).as_posix(), "sha256": file_sha256(child)})
        kind = "directory"
    else:
        raise ContractError("Authority must be a regular file or directory")
    return {"kind": kind, "files": files, "sha256": digest(files)}


def capture_authority(path: Path | str) -> dict:
    # Inspect the supplied path before resolve so symlinks are not silently accepted.
    captured = inventory(path)
    return {"path": str(Path(path).resolve()), "inventory": captured}


def safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ContractError("Artifact paths must be normalized relative paths")
    return path


def under(root: Path | str, relative: str) -> Path:
    root = Path(root).resolve()
    candidate = root / safe_relative(relative)
    if not candidate.resolve().is_relative_to(root):
        raise ContractError("Path escapes its declared root")
    return candidate


def _overlap(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def validate_contract(contract: dict) -> dict:
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("Unsupported contract schema")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", contract.get("work_id", "")):
        raise ContractError("Invalid work_id")
    roots = {}
    for name in ("work_root", "staging_root", "delivery_root"):
        raw = contract.get(name, "")
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise ContractError(f"{name} must be absolute")
        roots[name] = Path(raw).resolve()
    if _overlap(roots["staging_root"], roots["delivery_root"]) or _overlap(roots["work_root"], roots["delivery_root"]):
        raise ContractError("Delivery must be disjoint from work and staging")
    if roots["staging_root"] == roots["work_root"] or roots["work_root"].is_relative_to(roots["staging_root"]):
        raise ContractError("Staging cannot contain work state")
    authorities = contract.get("authorities", {})
    if not authorities:
        raise ContractError("Authorities are required")
    for authority in authorities.values():
        path = Path(authority["path"])
        if not path.is_absolute():
            raise ContractError("Authority paths must be absolute")
        for root in roots.values():
            if _overlap(path.resolve(), root):
                raise ContractError("Authorities must be disjoint from runtime/output roots")
        captured = authority["inventory"]
        if captured["kind"] not in {"file", "directory"} or digest(captured["files"]) != captured["sha256"]:
            raise ContractError("Invalid authority inventory")
        names = [item["path"] for item in captured["files"]]
        if names != sorted(set(names)):
            raise ContractError("Inventory must be unique and sorted")
    artifacts = contract.get("artifacts", [])
    ids, names = set(), set()
    for artifact in artifacts:
        name, aid = artifact["output"], artifact["id"]
        safe_relative(name)
        if not name.endswith(".xlsx") or aid in ids or name in names:
            raise ContractError("Unique .xlsx artifact outputs and IDs required")
        ids.add(aid); names.add(name)
        if set(artifact.get("authorities", {})) != {"values", "structure", "formatting"}:
            raise ContractError("Declare separate values, structure and formatting authorities")
        for reference in artifact["authorities"].values():
            authority_file(contract, reference)
        seen = set()
        for change in artifact.get("allowed_diffs", []):
            if set(change) != {"axis", "path", "before", "after"} or change["axis"] not in AXES:
                raise ContractError("Diffs require axis, exact path, before and after")
            if not change["path"] or not all(isinstance(x, str) for x in change["path"]):
                raise ContractError("Diff path must be a non-empty string list")
            key = (change["axis"], canonical(change["path"]))
            if key in seen:
                raise ContractError("Overlapping/duplicate diff declaration")
            seen.add(key)
            for slot in (change["before"], change["after"]):
                if slot != {"present": False} and (set(slot) != {"present", "value"} or slot["present"] is not True):
                    raise ContractError("Diff slots require present/value or present=false")
            if change["before"] == change["after"]:
                raise ContractError("Declared diff must actually change a property")
    if not artifacts:
        raise ContractError("At least one artifact is required")
    noop = contract.get("no_op_files", [])
    if len(noop) != len(set(noop)) or not set(noop) <= ids:
        raise ContractError("Invalid no_op_files")
    for artifact in artifacts:
        if artifact["id"] in noop:
            refs = list(artifact["authorities"].values())
            if artifact.get("allowed_diffs") or any(ref != refs[0] for ref in refs[1:]):
                raise ContractError("No-op requires one baseline and zero allowed diffs")
    policy = contract.get("recovery", {})
    for key, default, upper in (("max_attempts", 3, 10), ("same_failure_limit", 2, 10)):
        value = policy.get(key, default)
        if type(value) is not int or not 1 <= value <= upper:
            raise ContractError(f"Invalid recovery {key}")
    writer = contract.get("writer", {})
    if not writer.get("id") or not writer.get("version") or writer.get("cost") != "local":
        raise ContractError("This runner only accepts versioned local, no-API writers")
    canonical(contract)
    return contract


def authority_file(contract: dict, reference: dict) -> Path:
    try:
        authority = contract["authorities"][reference["authority"]]
        root = Path(authority["path"])
        relative = reference["file"]
        names = {entry["path"] for entry in authority["inventory"]["files"]}
        if relative not in names:
            raise ContractError("Referenced workbook missing from captured inventory")
        return root if authority["inventory"]["kind"] == "file" and relative == "." else under(root, relative)
    except (KeyError, TypeError) as error:
        raise ContractError("Invalid authority reference") from error


def check_inputs(contract: dict) -> dict:
    result = {}
    for name, authority in contract["authorities"].items():
        try:
            current = inventory(authority["path"])
        except (OSError, ContractError) as error:
            raise InputDrift(f"Authority unavailable: {name}") from error
        if current != authority["inventory"]:
            raise InputDrift(f"Authority changed: {name}")
        result[name] = current
    return result


def load_contract(path: Path | str) -> dict:
    return validate_contract(json.loads(Path(path).read_text(encoding="utf-8")))


def save_contract(path: Path | str, contract: dict) -> str:
    validate_contract(contract)
    check_inputs(contract)
    path = Path(path)
    if path.exists():
        raise ContractError("Existing contract is immutable; create a new work_id")
    atomic_json(path, contract)
    return file_sha256(path)
