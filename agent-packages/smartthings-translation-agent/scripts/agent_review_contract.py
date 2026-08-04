#!/usr/bin/env python3
"""Contracts for the read-only, sheet-level agent review workflow.

This module deliberately does not call an LLM.  It creates the evidence packet
that a lead agent and its specialist subagents must share, enforces the semantic
RAG allowance, and merges their structured opinions conservatively.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


SPECIALIST_ROLES = (
    "grammar_fluency",
    "semantic_fidelity",
    "localization_tone",
    "style_and_hard_rule_exceptions",
    "story_and_ui_coherence",
)


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


def validate_opinion(opinion: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "role": role, "sheet": sheet, "cell": cell, "finding_id": finding_id,
        "stance": stance, "reason": str(opinion.get("reason", "")).strip(),
        "after": opinion.get("after"), "rule_ids": list(opinion.get("rule_ids", [])),
    }


def merge_subjective_opinions(opinions: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Return proposals and human-review items using the two-role/no-opposition gate."""
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for raw in opinions:
        item = validate_opinion(raw)
        key = (item["sheet"], item["cell"], item["finding_id"], str(item["after"]))
        groups.setdefault(key, []).append(item)

    proposals, queue = [], []
    for (sheet, cell, finding_id, _after), items in groups.items():
        supporters = {item["role"] for item in items if item["stance"] == "support"}
        opponents = [item for item in items if item["stance"] == "oppose"]
        if len(supporters) >= 2 and not opponents and items[0]["after"]:
            proposals.append({
                "finding_id": finding_id, "sheet": sheet, "cell": cell,
                "after": items[0]["after"],
                "rule_ids": sorted({rule for item in items for rule in item["rule_ids"]}),
                "reason": "\n".join(item["reason"] for item in items if item["reason"]),
                "supporting_roles": sorted(supporters),
            })
        else:
            queue.append({
                "finding_id": finding_id, "sheet": sheet, "cell": cell,
                "reason": "specialist_disagreement_or_insufficient_support",
                "opinions": items,
            })
    return proposals, queue
