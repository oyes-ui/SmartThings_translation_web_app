#!/usr/bin/env python3
"""Contracts for the read-only, sheet-level agent review workflow.

This module deliberately does not call an LLM.  It creates the evidence packet
that a lead agent and its specialist subagents must share, enforces the semantic
RAG allowance, and merges their structured opinions conservatively.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from itertools import combinations
from threading import Lock
from typing import Any

__all__ = [
    "SPECIALIST_ROLES", "ROW_TYPES", "SemanticRagBudget", "ReviewMergeResult",
    "row_type_for_cell", "validate_opinion", "merge_subjective_opinions",
    "detect_role_anchoring", "echo_map", "independent_supporters",
    "CLEAN_STOP_REASONS", "RUN_METADATA_FIELDS",
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
    resolver_gate: str = "not_run"
    stage_reviews: dict[str, Any] = field(default_factory=dict)
    anchoring_metrics: dict[str, Any] = field(default_factory=dict)
    cell_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)


# Only these mean the specialist finished on its own terms; a truncated or errored
# run is as unusable as a missing one, so it is treated the same way.
CLEAN_STOP_REASONS = {"complete", "completed", "end_turn", "stop", "no_findings"}
RUN_METADATA_FIELDS = ("run_id", "agent", "model", "started_at", "finished_at",
                       "stop_reason", "confidence", "rag_evidence_ids", "error")


def _run_metadata(run: dict[str, Any] | None) -> dict[str, Any]:
    """Keep provided execution facts only — never invent or default them."""
    if not isinstance(run, dict):
        return {}
    return {key: run[key] for key in RUN_METADATA_FIELDS if run.get(key) not in (None, "", [])}


def _agent_runs(opinions: list[dict[str, Any]], expected_roles: tuple[str, ...], packet_id: str,
                role_runs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Derive agent_runs from collected opinions; never from a manual claim."""
    runs, missing = [], []
    for role in expected_roles:
        role_opinions = [item for item in opinions if item["role"] == role]
        metadata = _run_metadata(role_runs.get(role))
        if not role_opinions and not metadata:
            missing.append(role)
            runs.append({"role": role, "status": "missing", "opinions": 0, "packet_id": packet_id})
            continue
        mismatched = [item for item in role_opinions if item["packet_id"] and item["packet_id"] != packet_id]
        stop_reason = str(metadata.get("stop_reason", "")).lower()
        if mismatched or (packet_id and role_opinions and not any(item["packet_id"] for item in role_opinions)):
            status = "packet_mismatch"
        elif metadata.get("error"):
            status = "run_error"
        elif stop_reason and stop_reason not in CLEAN_STOP_REASONS:
            status = "run_incomplete"
        elif not role_opinions:
            # A clean no-findings run legitimately has no opinion rows.  It counts
            # only when the role file supplied an explicit clean stop reason;
            # metadata without that completion signal remains incomplete.
            status = "completed" if stop_reason in CLEAN_STOP_REASONS else "run_incomplete"
        else:
            status = "completed"
        if status != "completed":
            missing.append(role)
        runs.append({"role": role, "status": status, "opinions": len(role_opinions),
                     "packet_id": packet_id, **metadata})
    return runs, missing


def merge_subjective_opinions(opinions: list[dict[str, Any]], *, packet: dict[str, Any] | None = None,
                              deterministic_proposals: list[dict[str, Any]] | None = None,
                              expected_roles: tuple[str, ...] = SPECIALIST_ROLES,
                              role_runs: dict[str, dict[str, Any]] | None = None,
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
    anchoring = detect_role_anchoring(validated)
    echoes = echo_map(anchoring)

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
        independent = independent_supporters(supporters, echoes)
        if len(supporters) >= 2 and not opponents and items[0]["after"] and len(independent) < 2:
            # Two roles agreed, but one of them only repeats the other, so this is a
            # single opinion wearing two hats.  Ask for a genuinely independent
            # third perspective instead of rejecting a possibly correct fix.
            queue.append({**common, "reason": "anchored_support_needs_independent_role",
                          "supporting_roles": sorted(supporters),
                          "independent_supporting_roles": sorted(independent),
                          "opinions": items})
        elif len(independent) >= 2 and not opponents and items[0]["after"]:
            proposals.append({
                **common,
                "after": items[0]["after"],
                "rule_ids": sorted({rule for item in items for rule in item["rule_ids"]}),
                "reason": "\n".join(item["reason"] for item in items if item["reason"]),
                "supporting_roles": sorted(supporters),
                "independent_supporting_roles": sorted(independent),
                "origin": "subjective_consensus",
            })
        else:
            queue.append({**common, "reason": "specialist_disagreement_or_insufficient_support",
                          "opinions": items})

    runs, missing = _agent_runs(validated, expected_roles, packet_id, role_runs or {})
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
        anchoring=anchoring,
    )


def apply_resolver_gate(merged: ReviewMergeResult, validate, *, activation_candidates=None,
                        fail_closed: bool = False, strict_reasons: bool = False) -> ReviewMergeResult:
    """Re-check every proposed text against the deterministic card before it ships.

    The specialists reported their own ``constraint_status``; that is self-attestation,
    not verification, and a fluent model is at its most convincing exactly when it has
    quietly replaced a settled glossary target.  The app never trusts its own model
    here either — checker_service repairs and re-validates model output before letting
    it out — so the same floor applies to a proposal that reached consensus.

    ``validate(cell, after) -> dict`` is supplied by the caller (wired to
    GlossaryChecker.validate_audit_suggestion) so this module stays free of the app.
    A repaired text replaces the proposal; anything the resolver still refuses goes to
    the human queue rather than being rewritten into compliance.
    """
    candidates = {(str(item.get("cell", "")).upper(), str(item.get("source_term", "")))
                  for item in (activation_candidates or [])
                  if isinstance(item, dict) and not item.get("confirmed")}
    kept, queued = [], list(merged.human_review_queue)
    blocked_count = 0
    for proposal in merged.proposals:
        verdict = validate(proposal.get("cell", ""), proposal.get("after", ""))
        if not verdict:
            if fail_closed:
                blocked_count += 1
                queued.append({
                    "finding_id": proposal.get("finding_id", ""), "sheet": proposal.get("sheet", ""),
                    "cell": proposal.get("cell", ""), "row_type": proposal.get("row_type", ""),
                    "reason": "missing_constraint_evidence", "rejected_after": proposal.get("after", ""),
                    "resolver_status": "blocked", "resolver_violations": [], "opinions": [],
                })
            else:
                kept.append(proposal)
            continue
        if verdict.get("blocked") or verdict.get("status") not in {"pass", None}:
            blocked_count += 1
            violations = list(verdict.get("violations") or [])
            review = list(verdict.get("review") or [])
            cell = str(proposal.get("cell", "")).upper()
            activation_hits = [item for item in violations
                               if item.get("reason") == "missing_glossary_target"
                               and (cell, str(item.get("source_term", ""))) in candidates]
            activation_only = bool(activation_hits) and len(activation_hits) == len(violations) and not review
            if activation_only:
                reason, resolver_status = "glossary_activation_review", "human_review"
            elif strict_reasons:
                reason, resolver_status = "invalidated_by_hard_constraint", verdict.get("status", "blocked")
            else:
                reason, resolver_status = "blocked_by_resolver_revalidation", verdict.get("status", "")
            queued.append({
                "finding_id": proposal.get("finding_id", ""), "sheet": proposal.get("sheet", ""),
                "cell": proposal.get("cell", ""), "row_type": proposal.get("row_type", ""),
                "reason": reason, "resolver_status": resolver_status,
                "resolver_violations": violations or review,
                "activation_candidates": activation_hits,
                "rejected_after": proposal.get("after", ""),
                "opinions": [],
            })
            continue
        normalized = verdict.get("normalized", proposal.get("after", ""))
        if normalized != proposal.get("after"):
            # A wrapper or letter case the resolver owns; repaired exactly as the app
            # repairs its own model output, not treated as a disagreement.
            proposal = {**proposal, "after": normalized, "resolver_repaired": True}
        kept.append(proposal)
    status = f"ran; {len(kept)} passed, {blocked_count} blocked"
    return replace(merged, proposals=kept, human_review_queue=queued, resolver_gate=status)


def detect_role_anchoring(opinions: list[dict[str, Any]], *, minimum: int = 2,
                          partial_threshold: float = 0.8) -> list[dict[str, Any]]:
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

    def echoes(own: set, reference: set) -> bool:
        """Does `own` add little beyond `reference`?

        A strict subset adds nothing.  A near-copy adds almost nothing, and is
        included so that withholding one finding cannot evade the check.
        """
        if len(own) < minimum:
            return False
        return not (own - reference) or len(own & reference) / len(own) >= partial_threshold

    findings = []
    # Every qualifying pair is reported: one role can lean on several others, and
    # suppressing the second relation would let its support count as independent
    # wherever the first role happens not to be supporting.
    for role, other in combinations(sorted(by_role), 2):
        own, reference = by_role[role], by_role[other]
        forward, backward = echoes(own, reference), echoes(reference, own)
        if not forward and not backward:
            continue
        matched = len(own & reference)
        if forward and backward:
            # Each is a near-copy of the other, so neither can be shown to be the
            # original; both are discounted rather than picking a side.
            findings.append({
                "role": role, "echoes_role": other, "matched": matched,
                "independent_supports": len(own - reference), "direction": "undetermined",
                "detail": f"{role}와 {other}의 support가 {matched}건 겹쳐 서로의 사본에 가깝다 — "
                          f"어느 쪽이 먼저인지 판별 불가",
            })
            continue
        echo, source = (role, other) if forward else (other, role)
        independent = len(by_role[echo] - by_role[source])
        findings.append({
            "role": echo, "echoes_role": source, "matched": matched,
            "independent_supports": independent,
            "direction": "subset" if not independent else "partial",
            "detail": f"{echo}의 support {len(by_role[echo])}건 중 {matched}건이 {source}와 "
                      f"(finding_id, after) 일치 — 독립 지지 {independent}건",
        })
    return findings


def echo_map(anchoring: list[dict[str, Any]]) -> dict[str, set[str]]:
    """role -> the roles it echoes, used to discount non-independent support."""
    mapping: dict[str, set[str]] = {}
    for flag in anchoring:
        role, source = flag.get("role"), flag.get("echoes_role")
        if not role or not source:
            continue
        mapping.setdefault(role, set()).add(source)
        if flag.get("direction") == "undetermined":
            # Neither side can be shown to be the original, so each discounts the other.
            mapping.setdefault(source, set()).add(role)
    return mapping


def independent_supporters(supporters: set[str], echoes: dict[str, set[str]]) -> set[str]:
    """Drop a supporter whose agreement merely repeats another supporter here."""
    return {role for role in supporters if not (echoes.get(role, set()) & supporters)}
