"""Independent on-disk contract verification; never trusts generator expectations."""
from __future__ import annotations

import json
import hashlib
import warnings
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
from xml.etree.ElementTree import canonicalize

import openpyxl
from workbook_contract import (AXES, ROLES, ContractError, authority_file, canonical,
    check_inputs, digest, file_sha256, load_contract)
from workbook_mutation_guard import detailed_workbook_snapshot


def verifier_version() -> str:
    # Code identity invalidates records after changing serializer or verification rules.
    here = Path(__file__).parent
    return digest({"openpyxl": openpyxl.__version__, "code": {name: file_sha256(here / name) for name in
                   ("workbook_verifier.py", "workbook_contract.py", "workbook_mutation_guard.py")}})


class VerificationError(ValueError):
    code = "verification_failed"
    def __init__(self, report: dict):
        super().__init__("Workbook does not match the approved contract")
        self.report = report


def facts(path: Path | str) -> dict:
    path = Path(path)
    # Fail closed on objects that openpyxl cannot reliably preserve.
    extras = {}
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if any(token in name.lower() for token in
                   ("vba", "activex", "slicer", "pivot", "externallink", "embeddings/", "connections.xml")):
                raise ContractError(f"Unsupported OOXML object: {name}")
            if name.startswith(("xl/drawings/", "xl/charts/", "xl/media/")):
                data = archive.read(name)
                if name.endswith((".xml", ".rels")):
                    data = canonicalize(data.decode()).encode()
                extras[canonical(["package", name])] = hashlib.sha256(data).hexdigest()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always", UserWarning)
        book = openpyxl.load_workbook(path, rich_text=True, data_only=False)
    if any("not supported" in str(item.message).lower() or "will be removed" in str(item.message).lower()
           for item in notices):
        book.close()
        raise ContractError("Workbook contains unsupported content that openpyxl would discard")
    try:
        result = detailed_workbook_snapshot(book)
    finally:
        book.close()
    result["layout"].update(extras)
    return result


def slot(mapping: dict, key: str) -> dict:
    return {"present": True, "value": mapping[key]} if key in mapping else {"present": False}


def diff_facts(before: dict, after: dict) -> list[dict]:
    changes = []
    for axis in AXES:
        if digest(before[axis]) == digest(after[axis]):
            continue
        for key in sorted(set(before[axis]) | set(after[axis])):
            old, new = slot(before[axis], key), slot(after[axis], key)
            if old != new:
                changes.append({"axis": axis, "path": json.loads(key), "before": old, "after": new})
    return changes


def expected_baseline(contract: dict, artifact: dict) -> dict:
    cache = {}
    result = {}
    for axis in AXES:
        path = authority_file(contract, artifact["authorities"][ROLES[axis]])
        if path not in cache:
            cache[path] = facts(path)
        result[axis] = cache[path][axis]
    return result


def verify_artifact(contract_path: Path | str, artifact_id: str, staged: Path | str) -> dict:
    contract_path, staged = Path(contract_path), Path(staged)
    contract_hash = file_sha256(contract_path)
    contract = load_contract(contract_path)
    inputs = check_inputs(contract)
    artifact = next(a for a in contract["artifacts"] if a["id"] == artifact_id)
    staged_hash = file_sha256(staged)
    baseline = expected_baseline(contract, artifact)
    actual = facts(staged)
    observed = diff_facts(baseline, actual)
    allowed = artifact.get("allowed_diffs", [])
    for change in allowed:
        # Contract's before must itself agree with independently opened authorities.
        key = canonical(change["path"])
        if canonical(slot(baseline[change["axis"]], key)) != canonical(change["before"]):
            raise ContractError("Declared before disagrees with authority")
    actual_set, allowed_set = {canonical(x) for x in observed}, {canonical(x) for x in allowed}
    unexpected = [json.loads(x) for x in sorted(actual_set - allowed_set)]
    missing = [json.loads(x) for x in sorted(allowed_set - actual_set)]
    checks = {}
    for axis in AXES:
        checks[axis] = ("unexpected_diff" if any(x["axis"] == axis for x in unexpected)
                        else "missing_declared_diff" if any(x["axis"] == axis for x in missing)
                        else "passed")
    if (file_sha256(staged) != staged_hash or file_sha256(contract_path) != contract_hash
            or check_inputs(contract) != inputs):
        raise ContractError("Files changed during verification")
    report = {
        "status": "passed" if not unexpected and not missing else "failed",
        "artifact_id": artifact_id, "staged_file_sha256": staged_hash,
        "contract_sha256": contract_hash, "contract_content_sha256": digest(contract), "input_authority_sha256": inputs,
        "verifier_version": verifier_version(), "verified_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "unexpected": unexpected, "missing": missing,
        "observed_diff_count": len(observed),
        "axis_sha256": {axis: digest(actual[axis]) for axis in AXES},
    }
    if report["status"] != "passed":
        raise VerificationError(report)
    return report


def validate_record(contract_path: Path | str, artifact_id: str, path: Path | str, record: dict) -> None:
    contract = load_contract(contract_path)
    if (record.get("status") != "passed" or record.get("artifact_id") != artifact_id
            or record.get("contract_sha256") != file_sha256(contract_path)
            or record.get("contract_content_sha256") != digest(contract)
            or record.get("staged_file_sha256") != file_sha256(path)
            or record.get("verifier_version") != verifier_version()
            or record.get("checks") != {axis: "passed" for axis in AXES}
            or record.get("unexpected") != [] or record.get("missing") != []
            or record.get("input_authority_sha256") != check_inputs(contract)):
        raise ContractError("Stale, incomplete or mismatched verification record")


def validate_published_record(contract_path, artifact_id, path, record):
    """Authenticate an immutable historical result, not permission to publish anew.

    Retain the verifier and captured input identities from publication time.
    Staging/promotion must still use validate_record with current dependencies.
    """
    contract = load_contract(contract_path)
    if (record.get("status") != "passed" or record.get("artifact_id") != artifact_id
            or record.get("contract_sha256") != file_sha256(contract_path)
            or record.get("contract_content_sha256") != digest(contract)
            or record.get("staged_file_sha256") != file_sha256(path)
            or not isinstance(record.get("verifier_version"), str) or not record["verifier_version"]
            or record.get("checks") != {axis: "passed" for axis in AXES}
            or record.get("unexpected") != [] or record.get("missing") != []
            or record.get("input_authority_sha256") !=
                {name: authority["inventory"] for name, authority in contract["authorities"].items()}):
        raise ContractError("Incomplete or mismatched historical publication record")
