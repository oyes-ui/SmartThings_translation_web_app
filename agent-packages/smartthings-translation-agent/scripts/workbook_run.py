"""Small durable runner: local writer -> independent verify -> immutable batch publish.

No scheduling service or agent framework. The event journal is authoritative;
state.json is a replaceable read cache. OS locks expire when the process exits.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from workbook_contract import (ContractError, InputDrift, atomic_json, atomic_text,
    canonical, check_inputs, digest, file_sha256, load_contract, under)
from workbook_mutation_guard import save_verified_atomic
from workbook_verifier import VerificationError, validate_record, validate_published_record, verify_artifact


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ContractError("Run is already being executed by another process") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_state(work_root: Path | str) -> dict | None:
    journal = Path(work_root) / "events.jsonl"
    if not journal.exists():
        return None
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    if not events or [x["sequence"] for x in events] != list(range(1, len(events)+1)):
        raise ContractError("Invalid event journal")
    return events[-1]["state"]


def _checkpoint(root: Path, state: dict, event: str, **detail) -> None:
    journal = root / "events.jsonl"
    previous = journal.read_text() if journal.exists() else ""
    sequence = len(previous.splitlines()) + 1
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    entry = {"sequence": sequence, "event": event, "detail": detail,
             "at": state["updated_at"], "state": state}
    # Publish event first; a crash before the cache write is repaired on resume.
    atomic_text(journal, previous + canonical(entry) + "\n")
    atomic_json(root / "state.json", state)


def recovery_decision(error: Exception) -> dict:
    if isinstance(error, VerificationError):
        report = error.report
        affected = {x["axis"] for x in report["unexpected"] + report["missing"]}
        retry = bool(affected) and affected <= {"layout", "styles", "rich_text", "annotations"}
        return {"kind": "format_or_structure_mismatch" if retry else "value_mismatch",
                "action": "regenerate" if retry else "user_review",
                "signature": digest({"unexpected": report["unexpected"], "missing": report["missing"]})}
    if isinstance(error, OSError) and error.errno in {errno.EAGAIN, errno.EBUSY, errno.ETIMEDOUT}:
        return {"kind": "transient_io", "action": "regenerate", "signature": str(error.errno)}
    return {"kind": getattr(error, "code", "execution_error"), "action": "user_review",
            "signature": digest({"type": type(error).__name__, "message": str(error)})}


def _approved(approval_path: Path | str | None, contract_hash: str) -> bool:
    if approval_path is None:
        return False
    approval = json.loads(Path(approval_path).read_text())
    return (approval.get("approved") is True and approval.get("contract_sha256") == contract_hash
            and isinstance(approval.get("approved_by"), str) and bool(approval["approved_by"].strip()))


def _published(contract_path: Path, contract: dict) -> dict | None:
    target = under(contract["delivery_root"], contract["work_id"])
    if not target.exists():
        return None
    manifest_path = target / "batch.complete.json"
    if not manifest_path.is_file():
        raise ContractError("Existing publication has no completion manifest")
    manifest = json.loads(manifest_path.read_text())
    expected_outputs = {a["id"]: str(under(target, a["output"])) for a in contract["artifacts"]}
    if (manifest.get("status") != "completed" or manifest.get("work_id") != contract["work_id"]
            or manifest.get("outputs") != expected_outputs):
        raise ContractError("Publication metadata does not match the declared batch")
    if manifest.get("contract_sha256") != file_sha256(contract_path):
        raise ContractError("Existing publication belongs to another contract")
    if set(manifest.get("records", {})) != {a["id"] for a in contract["artifacts"]}:
        raise ContractError("Incomplete batch publication")
    for artifact in contract["artifacts"]:
        validate_published_record(contract_path, artifact["id"], under(target, artifact["output"]),
                        manifest["records"][artifact["id"]])
    return manifest


def promote_verified(contract_path: Path | str, records: dict) -> dict:
    """Publish one version directory atomically, after checking every copied artifact.

    Uses a destination-local temporary directory, so work and delivery may be on
    different volumes. Existing versions are never overwritten. Consumers must
    use the returned version directory, not scan hidden preparing directories.
    """
    contract_path = Path(contract_path)
    contract = load_contract(contract_path)
    root = Path(contract["delivery_root"])
    with exclusive_lock(root / ".publication.lock"):
        existing = _published(contract_path, contract)
        if existing is not None:
            return existing
        if set(records) != {a["id"] for a in contract["artifacts"]}:
            raise ContractError("Every artifact must be verified before promotion")
        for artifact in contract["artifacts"]:
            validate_record(contract_path, artifact["id"],
                            under(contract["staging_root"], artifact["output"]), records[artifact["id"]])
        temporary = Path(tempfile.mkdtemp(prefix=".preparing-" + contract["work_id"] + "-", dir=root))
        try:
            for artifact in contract["artifacts"]:
                copied = under(temporary, artifact["output"])
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(under(contract["staging_root"], artifact["output"]), copied)
                with copied.open("rb") as handle:
                    os.fsync(handle.fileno())
                validate_record(contract_path, artifact["id"], copied, records[artifact["id"]])
            target = under(root, contract["work_id"])
            manifest = {"status": "completed", "work_id": contract["work_id"],
                        "contract_sha256": file_sha256(contract_path), "records": records,
                        "outputs": {a["id"]: str(under(target, a["output"])) for a in contract["artifacts"]}}
            atomic_json(temporary / "batch.complete.json", manifest)
            # Recheck the originals and approval contract immediately before publication.
            check_inputs(contract)
            for artifact in contract["artifacts"]:
                validate_record(contract_path, artifact["id"], under(temporary, artifact["output"]), records[artifact["id"]])
            os.replace(temporary, target)
            directory = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def execute_run(contract_path: Path | str, build: Callable, *, writer: dict,
                approval_path: Path | str | None = None) -> dict:
    """build(contract, artifact, last_failure) returns a new in-memory workbook.

    It must read immutable authorities and perform no API calls. All workbook
    saves and delivery writes are owned by the runner. A fresh process can call
    this function again with the same files to resume.
    """
    contract_path = Path(contract_path).resolve()
    contract = load_contract(contract_path)
    root = Path(contract["work_root"])
    contract_hash = file_sha256(contract_path)
    with exclusive_lock(root / ".run.lock"):
        state = read_state(root)
        if state is None:
            staging = Path(contract["staging_root"])
            if staging.exists() and any(staging.iterdir()):
                raise ContractError("A new run requires an empty staging directory")
            state = {"status": "contracted", "contract_sha256": contract_hash,
                     "work_id": contract["work_id"], "attempts": {}, "failures": {},
                     "verified": [], "next_action": "approve_contract"}
            _checkpoint(root, state, "contract_loaded")
        if state["contract_sha256"] != contract_hash:
            raise ContractError("Contract changed; use a new run and new approval")
        atomic_json(root / "state.json", state)
        try:
            if not _approved(approval_path, contract_hash):
                state.update(status="awaiting_approval", next_action="approve_contract")
                _checkpoint(root, state, "approval_required")
                return state
            existing = _published(contract_path, contract)
            if existing is not None:
                state.update(status="completed", next_action=None, result=existing)
                state.pop("error", None)
                _checkpoint(root, state, "publication_reconciled")
                return state
            if state["status"] in {"blocked", "failed"}:
                return state
            if writer != contract["writer"]:
                raise ContractError("Writer version changed; create a new contract")
            check_inputs(contract)
            state.update(status="ready", next_action="generate_or_reverify")
            _checkpoint(root, state, "approval_validated")
            records = {}
            for artifact in contract["artifacts"]:
                aid = artifact["id"]
                staged = under(contract["staging_root"], artifact["output"])
                record_path = root / "verification" / (digest(aid) + ".json")
                must_generate = not staged.is_file()
                while True:
                    try:
                        if not must_generate:
                            state.update(status="verifying", next_action="verify_artifact")
                            _checkpoint(root, state, "reverify", artifact=aid)
                            # Reopening on resume is deliberate: no file-existence-only trust.
                            record = verify_artifact(contract_path, aid, staged)
                        else:
                            attempts = state["attempts"].get(aid, 0)
                            if attempts >= contract.get("recovery", {}).get("max_attempts", 3):
                                state.update(status="failed", next_action="inspect_failure_and_create_new_run")
                                _checkpoint(root, state, "retry_budget_exhausted", artifact=aid)
                                return state
                            state["attempts"][aid] = attempts + 1
                            state.update(status="generating", next_action="generate_artifact")
                            _checkpoint(root, state, "generation_started", artifact=aid, attempt=attempts+1)
                            book = build(contract, artifact, state["failures"].get(aid))
                            try:
                                state.update(status="verifying", next_action="verify_artifact")
                                _checkpoint(root, state, "verification_started", artifact=aid)
                                record = save_verified_atomic(book, staged,
                                    lambda candidate: verify_artifact(contract_path, aid, candidate),
                                    overwrite=True, contract=contract)
                            finally:
                                book.close()
                        atomic_json(record_path, record)
                        records[aid] = record
                        if aid not in state["verified"]:
                            state["verified"].append(aid)
                        state.update(status="verified", next_action="verify_remaining_or_publish")
                        _checkpoint(root, state, "artifact_verified", artifact=aid,
                                    artifact_sha256=record["staged_file_sha256"], verifier=record["verifier_version"])
                        break
                    except Exception as error:
                        decision = recovery_decision(error)
                        previous = state["failures"].get(aid, {})
                        repeats = previous.get("repeats", 0)+1 if previous.get("signature") == decision["signature"] else 1
                        failure = {**decision, "repeats": repeats, "message": str(error)}
                        state["failures"][aid] = failure
                        if isinstance(error, VerificationError):
                            atomic_json(root / "failures" / (digest(aid) + ".json"), error.report)
                        limited = (repeats >= contract.get("recovery", {}).get("same_failure_limit", 2)
                                   or state["attempts"].get(aid, 0) >= contract.get("recovery", {}).get("max_attempts", 3))
                        if decision["action"] != "regenerate" or limited:
                            state.update(status="failed" if limited else "blocked", next_action="inspect_failure_and_create_new_run")
                            _checkpoint(root, state, "recovery_stopped", artifact=aid, failure=failure)
                            return state
                        state.update(status="recovering", next_action="regenerate_from_authorities")
                        _checkpoint(root, state, "recovery_scheduled", artifact=aid, failure=failure)
                        must_generate = True
            state.update(status="promoting", next_action="publish_verified_batch")
            _checkpoint(root, state, "promotion_started")
            result = promote_verified(contract_path, records)
            state.update(status="completed", next_action=None, result=result)
            _checkpoint(root, state, "completed")
            return state
        except Exception as error:
            state.update(status="blocked", next_action="replan_and_reapprove" if isinstance(error, InputDrift)
                         else "inspect_failure_and_create_new_run",
                         error={"kind": getattr(error, "code", "execution_error"), "message": str(error)})
            _checkpoint(root, state, "blocked", error=state["error"])
            return state
