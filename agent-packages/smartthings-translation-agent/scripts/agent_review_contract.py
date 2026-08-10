#!/usr/bin/env python3
"""Contracts for the read-only, sheet-level agent review workflow.

This module deliberately does not call an LLM.  It creates the evidence packet
that a lead agent and its specialist subagents must share, enforces the semantic
RAG allowance, and merges their structured opinions conservatively.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from threading import Lock
from typing import Any

__all__ = [
    "SPECIALIST_ROLES", "ROW_TYPES", "SemanticRagBudget", "ReviewMergeResult",
    "row_type_for_cell", "validate_opinion", "merge_subjective_opinions",
    "detect_role_anchoring",
]


SPECIALIST_ROLES = (
    "grammar_fluency",
    "semantic_fidelity",
    "localization_tone",
    "style_and_hard_rule_exceptions",
    "story_and_ui_coherence",
)

# Content-row semantics shared by the packet builder and the merge: PromptBuilder's
# row-key policy, reused here so a finding can be grouped by content type as well
# as by role.  This repo's defects cluster by row type (039 regression criteria,
# the disclaimer prefix and nav-path bracket fixes), so the tag is what lets a
# later review ask whether the five-role split is the right axis at all.
ROW_TYPES = {
    7: "title", 8: "description", 10: "title", 11: "description", 12: "disclaimer", 13: "button",
    15: "title", 16: "description", 17: "disclaimer", 18: "button",
    20: "title", 21: "description", 22: "disclaimer", 23: "button",
    25: "title", 26: "description", 27: "disclaimer", 28: "button",
}


def row_type_for_cell(cell: str) -> str:
    """title/description/disclaimer/button for a content cell such as ``C11``."""
    match = re.fullmatch(r"[A-Z]+([0-9]+)", str(cell).strip().upper())
    return ROW_TYPES.get(int(match.group(1)), "") if match else ""


@dataclass
class SemanticRagBudget:
    """Thread-safe per-sheet semantic RAG allowance.

    A caller must reserve a call before querying.  Offline retrieval is not part
    of this budget and therefore is intentionally not represented here.
    """

    limit: int = 0
    used: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("semantic RAG budget은 0 이상이어야 합니다.")

    def reserve(self, evidence_id: str) -> bool:
        """Reserve one paid semantic lookup; never exceed the approved limit."""
        if not evidence_id:
            raise ValueError("semantic RAG evidence_id가 필요합니다.")
        with self._lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            self.evidence_ids.append(evidence_id)
            return True

    def report(self) -> dict[str, Any]:
        return {"semantic_budget": self.limit, "semantic_used": self.used,
                "semantic_evidence_ids": list(self.evidence_ids)}


def validate_opinion(opinion: dict[str, Any], *, require_constraint_verdict: bool = False) -> dict[str, Any]:
    """Validate an opinion returned by a specialist before lead aggregation."""
    role = opinion.get("role")
    if role not in SPECIALIST_ROLES:
        raise ValueError(f"알 수 없는 검수 관점: {role!r}")
    sheet = str(opinion.get("sheet", "")).strip()
    cell = str(opinion.get("cell", "")).strip().upper()
    finding_id = str(opinion.get("finding_id", "")).strip()
    stance = opinion.get("stance")
    if not sheet or not cell or not finding_id:
        raise ValueError("opinion에는 sheet, cell, finding_id가 필요합니다.")
    if stance not in {"support", "oppose", "review"}:
        raise ValueError("stance는 support, oppose, review 중 하나여야 합니다.")
    constraint_status = str(opinion.get("constraint_status", "not_checked"))
    if require_constraint_verdict and constraint_status not in {"pass", "human_review", "blocked"}:
        raise ValueError("병합 전 opinion에는 resolver constraint_status가 필요합니다.")
    return {
        "role": role, "sheet": sheet, "cell": cell, "finding_id": finding_id,
        "stance": stance, "reason": str(opinion.get("reason", "")).strip(),
        "after": opinion.get("after"), "rule_ids": list(opinion.get("rule_ids", [])),
        "constraint_status": constraint_status,
        "packet_id": str(opinion.get("packet_id", "")).strip(),
    }


@dataclass(frozen=True)
class ReviewMergeResult:
    """The only accepted input to review_report_builder.build_review_artifacts.

    Making this a distinct type — rather than a plain list — is the enforcement
    mechanism for §4-A: a lead agent cannot summarise specialist prose into
    ``changes[]`` because there is no way to hand the builder a free-form list.
    """

    proposals: list[dict[str, Any]]
    human_review_queue: list[dict[str, Any]]
    agent_runs: list[dict[str, Any]]
    sheet_status: str
    missing_roles: list[str]
    packet_id: str
    sheet: str
    cell_snapshot: dict[str, str] = field(default_factory=dict)
    anchoring: list[dict[str, Any]] = field(default_factory=list)


def _agent_runs(opinions: list[dict[str, Any]], expected_roles: tuple[str, ...],
                packet_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Derive agent_runs from collected opinions; never from a manual claim."""
    runs, missing = [], []
    for role in expected_roles:
        role_opinions = [item for item in opinions if item["role"] == role]
        if not role_opinions:
            missing.append(role)
            runs.append({"role": role, "status": "missing", "opinions": 0, "packet_id": packet_id})
            continue
        mismatched = [item for item in role_opinions if item["packet_id"] and item["packet_id"] != packet_id]
        if mismatched or (packet_id and not any(item["packet_id"] for item in role_opinions)):
            missing.append(role)
            status = "packet_mismatch"
        else:
            status = "completed"
        runs.append({"role": role, "status": status, "opinions": len(role_opinions),
                     "packet_id": packet_id})
    return runs, missing


def merge_subjective_opinions(opinions: list[dict[str, Any]], *, packet: dict[str, Any] | None = None,
                              deterministic_proposals: list[dict[str, Any]] | None = None,
                              expected_roles: tuple[str, ...] = SPECIALIST_ROLES,
                              enforce_constraints: bool = True) -> ReviewMergeResult:
    """Apply the two-role/no-opposition gate and report sheet completeness.

    ``packet`` ties every opinion to the evidence it was given.  A role that is
    absent, or answered against a different packet, makes the sheet
    ``incomplete``: no proposals are emitted at all, per §4-A.
    """
    packet = packet or {}
    packet_id = str(packet.get("packet_id", ""))
    sheet = str(packet.get("target_sheet", ""))
    validated = [validate_opinion(raw, require_constraint_verdict=enforce_constraints) for raw in opinions]

    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for item in validated:
        key = (item["sheet"], item["cell"], item["finding_id"], str(item["after"]))
        groups.setdefault(key, []).append(item)

    proposals, queue = [], []
    for (group_sheet, cell, finding_id, _after), items in groups.items():
        common = {"finding_id": finding_id, "sheet": group_sheet, "cell": cell,
                  "row_type": row_type_for_cell(cell)}
        constraint_statuses = {item["constraint_status"] for item in items}
        if "blocked" in constraint_statuses:
            queue.append({**common, "reason": "blocked_by_deterministic_constraint", "opinions": items})
            continue
        if "human_review" in constraint_statuses:
            queue.append({**common, "reason": "deterministic_constraint_requires_human_review",
                          "opinions": items})
            continue
        supporters = {item["role"] for item in items if item["stance"] == "support"}
        opponents = [item for item in items if item["stance"] == "oppose"]
        if len(supporters) >= 2 and not opponents and items[0]["after"]:
            proposals.append({
                **common,
                "after": items[0]["after"],
                "rule_ids": sorted({rule for item in items for rule in item["rule_ids"]}),
                "reason": "\n".join(item["reason"] for item in items if item["reason"]),
                "supporting_roles": sorted(supporters),
                "origin": "subjective_consensus",
            })
        else:
            queue.append({**common, "reason": "specialist_disagreement_or_insufficient_support",
                          "opinions": items})

    runs, missing = _agent_runs(validated, expected_roles, packet_id)
    sheet_status = "completed" if not missing else "incomplete"
    for proposal in deterministic_proposals or []:
        proposals.append({**proposal, "origin": "deterministic_hard_rule",
                          "row_type": row_type_for_cell(proposal.get("cell", ""))})

    if sheet_status == "incomplete":
        # §4-A: an incomplete sheet produces no changes at all — the candidates are
        # preserved for a human instead of being silently promoted.
        for proposal in proposals:
            queue.append({"finding_id": proposal.get("finding_id", ""), "sheet": proposal.get("sheet", sheet),
                          "cell": proposal.get("cell", ""), "row_type": proposal.get("row_type", ""),
                          "reason": "sheet_incomplete_missing_role_opinion",
                          "opinions": []})
        proposals = []

    return ReviewMergeResult(
        proposals=proposals, human_review_queue=queue, agent_runs=runs,
        sheet_status=sheet_status, missing_roles=missing, packet_id=packet_id,
        sheet=sheet, cell_snapshot=dict(packet.get("cell_snapshot", {})),
        anchoring=detect_role_anchoring(validated),
    )


def detect_role_anchoring(opinions: list[dict[str, Any]], *, minimum: int = 2) -> list[dict[str, Any]]:
    """Flag a role whose supporting opinions merely echo one other role.

    Per-finding text equality proves nothing here: the consensus gate groups by
    identical ``after``, so every promoted proposal is byte-identical by
    construction.  What is diagnostic is the *role* level — a specialist whose
    entire support set reproduces another single role's (finding_id, after)
    pairs contributed no independent evidence, which is exactly how
    build_story_ui_review.py built story_and_ui_coherence for the ES_CO batch.

    This reports; it never blocks.  Two roles can legitimately agree.
    """
    validated = [item if "stance" in item else validate_opinion(item) for item in opinions]
    by_role: dict[str, set[tuple[str, str]]] = {}
    for item in validated:
        if item["stance"] == "support" and item["after"] is not None:
            by_role.setdefault(item["role"], set()).add((item["finding_id"], str(item["after"])))

    findings, reported = [], set()
    for role, other in combinations(sorted(by_role), 2):
        pairs, other_pairs = by_role[role], by_role[other]
        if pairs == other_pairs and len(pairs) >= minimum:
            findings.append({
                "role": role, "echoes_role": other, "matched": len(pairs), "direction": "undetermined",
                "detail": f"{role}와 {other}의 support {len(pairs)}건이 완전히 동일 — 어느 쪽이 "
                          f"다른 쪽을 따랐는지 판별 불가, 독립 지지 0건",
            })
            reported |= {role, other}
            continue
        for echo, source in ((role, other), (other, role)):
            if echo in reported or len(by_role[echo]) < minimum or not by_role[echo] < by_role[source]:
                continue
            findings.append({
                "role": echo, "echoes_role": source, "matched": len(by_role[echo]), "direction": "subset",
                "detail": f"{echo}의 support {len(by_role[echo])}건이 모두 {source}와 "
                          f"(finding_id, after) 일치 — 독립 지지 0건",
            })
            reported.add(echo)
    return findings
