# -*- coding: utf-8 -*-
"""Load and validate the Markdown prompt rule files.

Rule content lives under ``rules/`` as YAML front matter plus a human-readable
body. The body is never parsed; the front matter is the only normative content.

- ``rules/languages/{canonical_key}.md`` -- per-language rules
- ``rules/bx_style.md`` -- Samsung BX voice
- ``rules/{common,typography,glossary,audit}.md`` -- flat rule documents whose
  items are positioned by ``slot``

A ``slot`` names a position in the assembled prompt. Python owns where a slot is
emitted and any branching between slots (e.g. picking the East-Asian navigation
quote rule); Markdown owns only the text.

Files are read once per process and cached. Validation failures raise
``RuleFileError`` at import time of :mod:`translation_web_app.prompt_builder`,
so a malformed rule file stops the app from starting rather than silently
producing prompts with missing rules.

Structural decisions -- alias resolution, fuzzy language matching, glossary
markers, bracket policy -- stay in Python; only rule content lives here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from translation_web_app.paths import APP_DIR, BX_STYLE_RULES_PATH, LANGUAGE_RULES_DIR, RULES_DIR

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = (1,)
SCOPES = ("app_prompt", "agent_audit", "excel_apply")
STATUSES = ("active", "draft", "deprecated")
SEVERITIES = ("blocker", "major", "minor", "info")
RULE_ORDERS = ("sequence",)
BX_GROUPS = ("OPEN", "BOLD", "AUTHENTIC", "NEGATIVE")

# Non-language, non-BX rule documents. Each is a flat rule list whose items are
# placed by `slot`, so one generic kind covers all of them instead of a bespoke
# dataclass + loader + RuleBundle field per category.
DOC_KINDS = ("common", "typography", "glossary", "audit")

# Slots a rule can occupy. The Python side owns *where* a slot is emitted and any
# branching between slots; the Markdown owns the text.
DOC_SLOTS = {
    "common": ("standard",),
    "typography": ("rule",),
    "glossary": (
        "term_rule", "bracket_precedence", "no_bracket", "bracket_wrap",
        "nav_exception", "nav_quote_default", "nav_quote_east_asian",
        "disclaimer_linebreak",
    ),
    "audit": ("intro", "checklist", "grade"),
}

# Slots whose text is fed through str.format() and therefore must keep their
# placeholders. Validated at load time: a missing placeholder would otherwise
# silently emit a literal "{open}" into a prompt.
SLOT_REQUIRED_PLACEHOLDERS = {"bracket_wrap": ("open", "close")}

_LANGUAGE_FIELDS = {
    "schema_version", "kind", "canonical_key", "display_name",
    "locale", "rule_order", "rules", "notes",
}
_BX_FIELDS = {
    "schema_version", "kind", "canonical_key", "display_name", "rule_order",
    "identity", "rules", "examples", "voice_definitions", "notes",
}
_DOC_FIELDS = {
    "schema_version", "kind", "canonical_key", "display_name", "rule_order",
    "rules", "notes",
}
_RULE_FIELDS = {
    "rule_id", "text", "scope", "status", "severity", "locale", "examples",
    "group", "slot", "label",
}
_IDENTITY_FIELDS = ("role", "persona", "goal")

_FRONT_MATTER_RE = re.compile(r"\A---\n(?P<front>.*?)\n---(?:\n(?P<body>.*))?\Z", re.DOTALL)


class RuleFileError(RuntimeError):
    """Raised when one or more rule files fail validation."""


class _DuplicateKey(Exception):
    def __init__(self, key: Any, line: int) -> None:
        super().__init__(key)
        self.key = key
        self.line = line


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of silently
    keeping the last one, which would drop rules with no error."""


def _no_duplicates(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise _DuplicateKey(key, key_node.start_mark.line + 1)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    text: str
    scope: tuple[str, ...]
    status: str = "active"
    severity: str | None = None
    locale: str | None = None
    examples: tuple[str, ...] = ()
    group: str | None = None
    slot: str | None = None
    label: str | None = None

    def in_scope(self, scope: str) -> bool:
        return scope in self.scope and self.status == "active"


@dataclass(frozen=True)
class LanguageRules:
    canonical_key: str
    display_name: str
    schema_version: int
    source_path: Path
    rules: tuple[Rule, ...]
    body: str = ""
    locale: str | None = None

    def prompt_rules(self) -> tuple[str, ...]:
        """Rule text destined for the app prompt, in file order."""
        return tuple(rule.text for rule in self.rules if rule.in_scope("app_prompt"))

    def scoped(self, scope: str) -> tuple[Rule, ...]:
        return tuple(rule for rule in self.rules if rule.in_scope(scope))


@dataclass(frozen=True)
class BxStyle:
    role: str
    persona: str
    goal: str
    rules: tuple[Rule, ...]
    examples: tuple[Mapping[str, str], ...]
    source_path: Path
    body: str = ""
    voice_definitions: Mapping[str, str] = field(default_factory=dict)

    @property
    def voice_attributes(self) -> dict[str, tuple[str, ...]]:
        """OPEN/BOLD/AUTHENTIC -> rule text, in first-appearance order."""
        grouped: dict[str, list[str]] = {}
        for rule in self.rules:
            if rule.group and rule.group != "NEGATIVE" and rule.in_scope("app_prompt"):
                grouped.setdefault(rule.group, []).append(rule.text)
        return {name: tuple(items) for name, items in grouped.items()}

    @property
    def negative_constraints(self) -> tuple[str, ...]:
        return tuple(
            rule.text
            for rule in self.rules
            if rule.group == "NEGATIVE" and rule.in_scope("app_prompt")
        )

    @property
    def few_shot_examples(self) -> tuple[Mapping[str, str], ...]:
        return self.examples

    def as_legacy_dict(self) -> dict:
        """Reconstruct the legacy BX_STYLE_RULES shape, for equivalence tests."""
        voice: dict[str, dict] = {}
        for name, items in self.voice_attributes.items():
            entry: dict[str, Any] = {"actionable_rules": list(items)}
            if name in self.voice_definitions:
                entry["definition"] = self.voice_definitions[name]
            voice[name] = entry
        return {
            "system_identity": {"role": self.role, "persona": self.persona, "goal": self.goal},
            "voice_attributes": voice,
            "negative_constraints": list(self.negative_constraints),
            "few_shot_examples": [dict(item) for item in self.examples],
        }


@dataclass(frozen=True)
class RuleDoc:
    """A flat rule document (common / typography / glossary / audit)."""

    canonical_key: str
    display_name: str
    schema_version: int
    source_path: Path
    rules: tuple[Rule, ...]
    body: str = ""

    def slot(self, slot: str) -> tuple[Rule, ...]:
        """Active app_prompt rules in a slot, in file order."""
        return tuple(r for r in self.rules if r.slot == slot and r.in_scope("app_prompt"))

    def texts(self, slot: str) -> tuple[str, ...]:
        return tuple(r.text for r in self.slot(slot))

    def one(self, slot: str) -> str:
        """The single rule text in a slot; '' when absent.

        Replaces the old positional indexing (e.g. GLOSSARY_TERM_RULES['rules'][0]),
        so reordering a file can no longer silently change a prompt.
        """
        found = self.texts(slot)
        return found[0] if found else ""

    def labelled(self, slot: str) -> tuple[tuple[str, str], ...]:
        """(label, text) pairs — audit checklist categories and grade criteria."""
        return tuple((r.label or "", r.text) for r in self.slot(slot))


@dataclass(frozen=True)
class RuleBundle:
    languages: Mapping[str, LanguageRules]
    bx: BxStyle
    docs: Mapping[str, RuleDoc] = field(default_factory=dict)

    def doc(self, kind: str) -> RuleDoc:
        return self.docs[kind]

    def display_names(self) -> dict[str, str]:
        return {key: value.display_name for key, value in self.languages.items()}

    def prompt_rules_map(self) -> dict[str, list[str]]:
        return {key: list(value.prompt_rules()) for key, value in self.languages.items()}

    def rules_for_scope(self, scope: str) -> tuple[tuple[str, Rule], ...]:
        """Every rule in a scope, paired with its canonical key.

        This is how agent-side tooling reaches ``agent_audit`` rules, which are
        deliberately excluded from both the translation and audit prompts. Every
        rule source must be listed here or its rules become invisible to that
        tooling with no error.
        """
        found: list[tuple[str, Rule]] = [
            (key, rule)
            for key, language in self.languages.items()
            for rule in language.scoped(scope)
        ]
        found.extend(("bx_style", rule) for rule in self.bx.rules if rule.in_scope(scope))
        for kind, document in self.docs.items():
            found.extend((kind, rule) for rule in document.rules if rule.in_scope(scope))
        return tuple(found)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(APP_DIR.parent))
    except ValueError:
        return str(path)


def _split_front_matter(path: Path, errors: list[str]) -> tuple[dict | None, str]:
    raw = path.read_text(encoding="utf-8")
    text = raw.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        errors.append(
            f"{_rel(path)}: missing YAML front matter "
            "(file must start with a '---' line and contain a closing '---' line)"
        )
        return None, ""
    try:
        front = yaml.load(match.group("front"), Loader=_StrictLoader)
    except _DuplicateKey as exc:
        errors.append(
            f"{_rel(path)}: duplicate key {exc.key!r} in YAML front matter (line {exc.line})"
        )
        return None, ""
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        detail = getattr(exc, "problem", str(exc))
        where = f" at line {mark.line + 1}" if mark is not None else ""
        errors.append(f"{_rel(path)}: invalid YAML front matter{where}: {detail}")
        return None, ""
    if not isinstance(front, dict):
        errors.append(
            f"{_rel(path)}: front matter must be a YAML mapping, "
            f"got {type(front).__name__}"
        )
        return None, ""
    return front, (match.group("body") or "")


def _require_str(front: Mapping, name: str, path: Path, errors: list[str]) -> str | None:
    value = front.get(name)
    if name not in front:
        errors.append(f"{_rel(path)}: missing required field {name!r}")
        return None
    if not isinstance(value, str) or not value.strip():
        errors.append(
            f"{_rel(path)}: field {name!r} must be a non-empty string, "
            f"got {type(value).__name__}"
        )
        return None
    return value


def _check_common(front: Mapping, path: Path, kind: str, allowed: set[str], errors: list[str]) -> None:
    for name in sorted(set(front) - allowed):
        errors.append(
            f"{_rel(path)}: unknown front matter field {name!r} "
            f"(allowed: {', '.join(sorted(allowed))})"
        )
    version = front.get("schema_version")
    if "schema_version" not in front:
        errors.append(f"{_rel(path)}: missing required field 'schema_version'")
    elif version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            f"{_rel(path)}: unsupported schema_version {version!r} "
            f"(supported: {', '.join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)})"
        )
    if front.get("kind") != kind:
        errors.append(f"{_rel(path)}: kind must be {kind!r}, got {front.get('kind')!r}")
    order = front.get("rule_order", "sequence")
    if order not in RULE_ORDERS:
        errors.append(
            f"{_rel(path)}: rule_order {order!r} is not supported "
            f"(supported: {', '.join(RULE_ORDERS)})"
        )


def _parse_rules(
    front: Mapping,
    path: Path,
    errors: list[str],
    seen_rule_ids: dict[str, Path],
    groups: tuple[str, ...] | None = None,
    slots: tuple[str, ...] | None = None,
) -> tuple[Rule, ...]:
    """Parse the shared ``rules`` array.

    ``groups`` / ``slots`` are the per-kind vocabularies: when given, the
    corresponding field is required and validated against them; when None the
    field is rejected as unknown.
    """
    raw_rules = front.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        errors.append(
            f"{_rel(path)}: field 'rules' must be a non-empty list of mappings, "
            f"got {type(raw_rules).__name__}"
        )
        return ()

    parsed: list[Rule] = []
    for index, item in enumerate(raw_rules):
        where = f"rules[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{_rel(path)}: {where} must be a mapping, got {type(item).__name__}")
            continue

        allowed = set(_RULE_FIELDS)
        if groups is None:
            allowed.discard("group")
        if slots is None:
            allowed -= {"slot", "label"}
        for name in sorted(set(item) - allowed):
            errors.append(
                f"{_rel(path)}: {where} has unknown field {name!r} "
                f"(allowed: {', '.join(sorted(allowed))})"
            )

        rule_id = item.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            errors.append(f"{_rel(path)}: {where} missing required field 'rule_id'")
            continue
        label = f"{where} ({rule_id!r})"

        text = item.get("text")
        if "text" not in item:
            errors.append(f"{_rel(path)}: {label} missing required field 'text'")
            continue
        if not isinstance(text, str) or not text.strip():
            errors.append(
                f"{_rel(path)}: {where}.text must be a non-empty string, "
                f"got {type(text).__name__}"
            )
            continue

        if rule_id in seen_rule_ids:
            errors.append(
                f"{_rel(path)}: rule_id {rule_id!r} is already defined by "
                f"{_rel(seen_rule_ids[rule_id])}"
            )
        else:
            seen_rule_ids[rule_id] = path

        raw_scope = item.get("scope")
        if not isinstance(raw_scope, list) or not raw_scope or not all(
            isinstance(entry, str) for entry in raw_scope
        ):
            errors.append(
                f"{_rel(path)}: {where}.scope must be a non-empty list of strings, "
                f"got {type(raw_scope).__name__}"
            )
            continue
        bad_scope = False
        for position, entry in enumerate(raw_scope):
            if entry not in SCOPES:
                errors.append(
                    f"{_rel(path)}: {where}.scope[{position}] {entry!r} is not a "
                    f"supported scope ({', '.join(SCOPES)})"
                )
                bad_scope = True
        if bad_scope:
            continue

        status = item.get("status", "active")
        if status not in STATUSES:
            errors.append(
                f"{_rel(path)}: {where}.status {status!r} is not supported "
                f"({', '.join(STATUSES)})"
            )
            continue
        severity = item.get("severity")
        if severity is not None and severity not in SEVERITIES:
            errors.append(
                f"{_rel(path)}: {where}.severity {severity!r} is not supported "
                f"({', '.join(SEVERITIES)})"
            )
            continue
        raw_examples = item.get("examples", [])
        if not isinstance(raw_examples, list) or not all(
            isinstance(entry, str) for entry in raw_examples
        ):
            errors.append(
                f"{_rel(path)}: {where}.examples must be a list of strings, "
                f"got {type(raw_examples).__name__}"
            )
            continue
        group = item.get("group")
        if groups is not None and group not in groups:
            errors.append(
                f"{_rel(path)}: {where}.group {group!r} is not supported "
                f"({', '.join(groups)})"
            )
            continue

        slot = item.get("slot")
        if slots is not None:
            if slot not in slots:
                errors.append(
                    f"{_rel(path)}: {where}.slot {slot!r} is not supported "
                    f"({', '.join(slots)})"
                )
                continue
            missing = [
                name for name in SLOT_REQUIRED_PLACEHOLDERS.get(slot, ())
                if "{" + name + "}" not in text
            ]
            if missing:
                errors.append(
                    f"{_rel(path)}: {where}.text is missing required placeholder(s) "
                    f"{', '.join('{' + n + '}' for n in missing)} for slot {slot!r}"
                )
                continue

        parsed.append(
            Rule(
                rule_id=rule_id,
                text=text,
                scope=tuple(raw_scope),
                status=status,
                severity=severity,
                locale=item.get("locale"),
                examples=tuple(raw_examples),
                group=group if groups is not None else None,
                slot=slot if slots is not None else None,
                label=item.get("label") if slots is not None else None,
            )
        )
    return tuple(parsed)


def _load_language_file(
    path: Path, errors: list[str], seen_rule_ids: dict[str, Path]
) -> LanguageRules | None:
    front, body = _split_front_matter(path, errors)
    if front is None:
        return None
    _check_common(front, path, "language", _LANGUAGE_FIELDS, errors)
    canonical_key = _require_str(front, "canonical_key", path, errors)
    display_name = _require_str(front, "display_name", path, errors)
    rules = _parse_rules(front, path, errors, seen_rule_ids)
    if canonical_key is None or display_name is None:
        return None
    if canonical_key != path.stem:
        errors.append(
            f"{_rel(path)}: canonical_key {canonical_key!r} does not match "
            f"filename stem {path.stem!r}"
        )
        return None
    return LanguageRules(
        canonical_key=canonical_key,
        display_name=display_name,
        schema_version=front.get("schema_version", SCHEMA_VERSION),
        source_path=path,
        rules=rules,
        body=body,
        locale=front.get("locale"),
    )


def _load_doc_file(
    path: Path, kind: str, errors: list[str], seen_rule_ids: dict[str, Path]
) -> RuleDoc | None:
    """Load one of the flat rule documents (common/typography/glossary/audit)."""
    if not path.is_file():
        errors.append(f"{_rel(path)}: file not found")
        return None
    front, body = _split_front_matter(path, errors)
    if front is None:
        return None
    _check_common(front, path, kind, _DOC_FIELDS, errors)
    display_name = _require_str(front, "display_name", path, errors)
    rules = _parse_rules(front, path, errors, seen_rule_ids, slots=DOC_SLOTS[kind])
    if display_name is None:
        return None
    canonical_key = front.get("canonical_key")
    if canonical_key != kind:
        errors.append(
            f"{_rel(path)}: canonical_key {canonical_key!r} must equal the kind {kind!r}"
        )
        return None
    return RuleDoc(
        canonical_key=kind,
        display_name=display_name,
        schema_version=front.get("schema_version", SCHEMA_VERSION),
        source_path=path,
        rules=rules,
        body=body,
    )


def _load_bx_file(path: Path, errors: list[str], seen_rule_ids: dict[str, Path]) -> BxStyle | None:
    if not path.is_file():
        errors.append(f"{_rel(path)}: file not found")
        return None
    front, body = _split_front_matter(path, errors)
    if front is None:
        return None
    _check_common(front, path, "bx_style", _BX_FIELDS, errors)

    identity = front.get("identity")
    if not isinstance(identity, dict):
        errors.append(
            f"{_rel(path)}: field 'identity' must be a mapping with keys "
            f"{', '.join(_IDENTITY_FIELDS)}"
        )
        return None
    for name in _IDENTITY_FIELDS:
        if not isinstance(identity.get(name), str) or not identity[name].strip():
            errors.append(f"{_rel(path)}: identity is missing required key {name!r}")
            return None

    raw_examples = front.get("examples", [])
    if not isinstance(raw_examples, list):
        errors.append(
            f"{_rel(path)}: field 'examples' must be a list, got {type(raw_examples).__name__}"
        )
        return None
    examples: list[Mapping[str, str]] = []
    for index, item in enumerate(raw_examples):
        if not isinstance(item, dict) or set(item) != {"type", "input", "output"}:
            errors.append(
                f"{_rel(path)}: examples[{index}] must be a mapping with keys type, input, output"
            )
            return None
        examples.append({key: str(value) for key, value in item.items()})

    definitions = front.get("voice_definitions", {})
    if not isinstance(definitions, dict):
        errors.append(
            f"{_rel(path)}: field 'voice_definitions' must be a mapping, "
            f"got {type(definitions).__name__}"
        )
        return None

    rules = _parse_rules(front, path, errors, seen_rule_ids, groups=BX_GROUPS)
    return BxStyle(
        role=identity["role"],
        persona=identity["persona"],
        goal=identity["goal"],
        rules=rules,
        examples=tuple(examples),
        source_path=path,
        body=body,
        voice_definitions=dict(definitions),
    )


def load_rules(rules_dir: Path | str = RULES_DIR) -> RuleBundle:
    """Read and validate every rule file. Raises RuleFileError on any problem.

    Errors are aggregated across all files so a bad batch reports every issue at
    once rather than one per run.
    """
    rules_dir = Path(rules_dir)
    languages_dir = (
        LANGUAGE_RULES_DIR if rules_dir == RULES_DIR else rules_dir / "languages"
    )
    bx_path = BX_STYLE_RULES_PATH if rules_dir == RULES_DIR else rules_dir / "bx_style.md"

    errors: list[str] = []
    seen_rule_ids: dict[str, Path] = {}
    languages: dict[str, LanguageRules] = {}
    sources: dict[str, Path] = {}

    paths: Sequence[Path] = sorted(languages_dir.glob("*.md")) if languages_dir.is_dir() else []
    if not paths:
        errors.append(f"{_rel(languages_dir)}: no rule files found (expected at least one *.md)")
    for path in paths:
        language = _load_language_file(path, errors, seen_rule_ids)
        if language is None:
            continue
        if language.canonical_key in languages:
            errors.append(
                f"{_rel(path)}: canonical_key {language.canonical_key!r} is already "
                f"defined by {_rel(sources[language.canonical_key])}"
            )
            continue
        languages[language.canonical_key] = language
        sources[language.canonical_key] = path

    bx = _load_bx_file(bx_path, errors, seen_rule_ids)

    docs: dict[str, RuleDoc] = {}
    for kind in DOC_KINDS:
        document = _load_doc_file(rules_dir / f"{kind}.md", kind, errors, seen_rule_ids)
        if document is not None:
            docs[kind] = document

    # Reject stray *.md at the rules/ root. Without this an unexpected or
    # misspelled filename is silently ignored -- it fails open, so a typo'd
    # rule file would look installed while contributing nothing.
    expected = {"bx_style.md"} | {f"{kind}.md" for kind in DOC_KINDS}
    for path in sorted(rules_dir.glob("*.md")) if rules_dir.is_dir() else []:
        if path.name not in expected:
            errors.append(
                f"{_rel(path)}: unexpected rule file (expected one of "
                f"{', '.join(sorted(expected))}, or a file under languages/)"
            )

    if errors:
        detail = "\n".join(f"  - {message}" for message in errors)
        raise RuleFileError(f"{len(errors)} problem(s) in {_rel(rules_dir)}:\n{detail}")
    assert bx is not None  # guaranteed: a missing/invalid bx file appends an error
    return RuleBundle(languages=languages, bx=bx, docs=docs)


@lru_cache(maxsize=1)
def get_rules() -> RuleBundle:
    """Process-wide cached rule bundle. Loaded once; no hot reload by design."""
    return load_rules()
