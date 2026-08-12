"""Deterministic glossary/section constraint resolution shared by app and agents.

Lexical glossary requirements and bracket rendering are deliberately separate:
the term target/casing remains fixed, while ``no_bracket`` is composed from the
row role, navigation-path context, and a glossary exemption.  RAG/LLM output
is validated against this result; it never changes the result itself.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from pathlib import Path
from typing import Any, Callable


def _normalise_term(value: str) -> str:
    return str(value or "").strip().casefold()


def term_occurrence_pattern(term: str) -> str:
    """Match a glossary target as a whole token, not as a substring.

    ``Rutina`` must not match inside ``Rutinas`` — Spanish and Portuguese
    pluralise by suffix, so a bare substring search reports the plural as a
    casing violation of the singular.  The alphanumeric-only guard is what the
    casing repair already uses; keeping the two identical is the point, because
    a checker stricter than its own repair flags text nothing can fix.
    """
    pattern = re.escape(term)
    if term[:1].isalnum():
        pattern = r"(?<![a-zA-Z0-9])" + pattern
    if term[-1:].isalnum():
        pattern = pattern + r"(?![a-zA-Z0-9])"
    return pattern


@dataclass(frozen=True)
class TermConstraint:
    source_term: str
    target: str
    active: bool
    activation_source: str
    rule_ids: tuple[str, ...]
    bracket_policy: str  # inactive | wrap | no_bracket | blocked_rule_conflict
    no_bracket_reasons: tuple[str, ...]
    blocked_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class OccurrenceActivationManifest:
    """Read a story/cell/source-term activation manifest without changing it."""

    def __init__(self, entries: list[dict[str, Any]] | None = None):
        self._entries: dict[tuple[str, str, str], dict[str, Any]] = {}
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            story = str(entry.get("story", "")).zfill(3)
            cell = str(entry.get("cell", "")).upper()
            term = _normalise_term(entry.get("source_term", ""))
            if story and cell and term:
                self._entries[(story, cell, term)] = entry

    @classmethod
    def from_file(cls, path: str | Path | None) -> "OccurrenceActivationManifest":
        if not path:
            return cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
            raise ValueError("activation manifest에는 entries 배열이 필요합니다.")
        return cls(payload["entries"])

    def lookup(self, story: str | None, cell: str | None, source_term: str) -> dict[str, Any] | None:
        if not story or not cell:
            return None
        return self._entries.get((str(story).zfill(3), str(cell).upper(), _normalise_term(source_term)))


class ConstraintResolver:
    """Resolve and validate deterministic constraints for one source/target cell."""

    def __init__(
        self,
        *,
        glossary: dict[str, dict[str, Any]],
        get_relevant_terms: Callable[[str], list[str]],
        get_target: Callable[[dict[str, Any], str], str | None],
        is_deactivated: Callable[[str], bool],
        has_exempt_marker: Callable[[str], bool],
        get_context_mode: Callable[[str], str],
        activation_manifest: OccurrenceActivationManifest | None = None,
    ):
        self.glossary = glossary
        self.get_relevant_terms = get_relevant_terms
        self.get_target = get_target
        self.is_deactivated = is_deactivated
        self.has_exempt_marker = has_exempt_marker
        self.get_context_mode = get_context_mode
        self.activation_manifest = activation_manifest or OccurrenceActivationManifest()

    @staticmethod
    def _clean_target(value: str) -> str:
        return re.sub(r"\(.*?\)", "", value or "").strip()

    def resolve(
        self,
        source_text: str,
        target_lang_code: str,
        *,
        row_key: str = "",
        story: str | None = None,
        cell: str | None = None,
        inside_navigation_path: bool = False,
    ) -> list[TermConstraint]:
        context_mode = self.get_context_mode(row_key)
        result: list[TermConstraint] = []
        for term in self.get_relevant_terms(source_text) or []:
            meta = self.glossary.get(term)
            if not meta:
                continue
            target = self._clean_target(self.get_target(meta.get("targets", {}), target_lang_code) or "")
            if not target:
                continue
            rule = str(meta.get("rule", ""))
            entry = self.activation_manifest.lookup(story, cell, term)
            active = bool(entry.get("active")) if entry is not None else not self.is_deactivated(rule)
            activation_source = "occurrence_manifest" if entry is not None else "glossary_fallback"
            rule_ids = ["glossary-target", "glossary-casing"]
            if entry is not None:
                rule_ids.append("occurrence-activation")
            if not active:
                result.append(TermConstraint(term, target, False, activation_source, tuple(rule_ids), "inactive", ()))
                continue

            no_bracket_reasons: list[str] = []
            if context_mode == "title_button":
                no_bracket_reasons.append("section_role")
            if inside_navigation_path:
                no_bracket_reasons.append("navigation_path")
            if self.has_exempt_marker(rule):
                no_bracket_reasons.append("glossary_exempt")
            rule_ids.append("bracket-policy")

            # Reserved future syntax: explicit must-wrap must never silently
            # override a structural no-bracket policy.
            explicit_must_wrap = "must_wrap" in rule.lower().replace(" ", "")
            if explicit_must_wrap and no_bracket_reasons:
                result.append(TermConstraint(
                    term, target, True, activation_source, tuple(rule_ids),
                    "blocked_rule_conflict", tuple(no_bracket_reasons), "must_wrap_conflicts_with_no_bracket",
                ))
            else:
                policy = "no_bracket" if no_bracket_reasons else "wrap"
                result.append(TermConstraint(term, target, True, activation_source, tuple(rule_ids), policy, tuple(no_bracket_reasons)))
        return result

    def card(self, constraints: list[TermConstraint]) -> dict[str, Any]:
        return {
            "authority": "deterministic_hard_constraints",
            "rag_policy": "advisory_only_cannot_override_constraints",
            "terms": [item.as_dict() for item in constraints],
        }

    def validate_target(self, proposed_text: str, constraints: list[TermConstraint], *,
                        navigation_spans: list[tuple[int, int]] | None = None) -> dict[str, Any]:
        """Return pass/blocked/human_review; never rewrite the proposed text."""
        blocked: list[dict[str, Any]] = []
        review: list[dict[str, Any]] = []
        # Longest target first, so a term contained in a longer one is judged as part
        # of it rather than on its own.  `Kia` occurs inside `BlueLink∙KIA Connect`;
        # judged separately it reads as a casing violation of correct text.
        ordered = sorted(constraints, key=lambda item: len(item.target or ""), reverse=True)
        claimed: list[tuple[int, int]] = []
        for item in ordered:
            if item.bracket_policy == "blocked_rule_conflict":
                review.append({"source_term": item.source_term, "reason": item.blocked_reason, "rule_ids": item.rule_ids})
                continue
            if not item.active:
                continue
            found = list(re.finditer(term_occurrence_pattern(item.target), proposed_text, re.IGNORECASE))
            occurrences = [match for match in found
                           if not any(start <= match.start() and match.end() <= end for start, end in claimed)]
            claimed.extend((match.start(), match.end()) for match in occurrences)
            if not occurrences:
                if found:
                    # Present, but only inside a longer glossary term that already
                    # satisfies it — not a missing target.
                    continue
                blocked.append({"source_term": item.source_term, "reason": "missing_glossary_target", "expected": item.target, "rule_ids": item.rule_ids})
                continue
            for occurrence in occurrences:
                actual = proposed_text[occurrence.start():occurrence.end()]
                if actual != item.target:
                    blocked.append({"source_term": item.source_term, "reason": "glossary_casing_mismatch", "expected": item.target, "actual": actual, "rule_ids": item.rule_ids})
                wrapped = occurrence.start() > 0 and occurrence.end() < len(proposed_text) and proposed_text[occurrence.start() - 1] == "[" and proposed_text[occurrence.end()] == "]"
                inside_navigation_path = any(start <= occurrence.start() and occurrence.end() <= end for start, end in (navigation_spans or []))
                effective_policy = "no_bracket" if inside_navigation_path else item.bracket_policy
                extra_reasons = ("navigation_path",) if inside_navigation_path else ()
                effective_reasons = tuple(dict.fromkeys(item.no_bracket_reasons + extra_reasons))
                if effective_policy == "no_bracket" and wrapped:
                    blocked.append({"source_term": item.source_term, "reason": "forbidden_bracket", "reasons": effective_reasons, "rule_ids": item.rule_ids})
                elif effective_policy == "wrap" and not wrapped:
                    review.append({"source_term": item.source_term, "reason": "missing_required_bracket", "rule_ids": item.rule_ids})
        status = "blocked" if blocked else "human_review" if review else "pass"
        return {"status": status, "blocked": blocked, "review": review, "constraints": [item.as_dict() for item in constraints]}
