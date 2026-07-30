#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seed src/translation_web_app/rules/ from the legacy prompt_modules constants.

One-shot migration tool.  Run once to create the Markdown rule files from
``LANGUAGE_LOCALIZATION_RULES``, ``LANGUAGE_RULE_LABELS`` and ``BX_STYLE_RULES``.
After the first hand edit (a new locale, or any ``agent_audit`` rule) ``--check``
reports a legitimate mismatch and this script must not be re-run: it would
clobber hand-authored content.

Imports only ``prompt_modules`` -- never ``prompt_builder``, which loads the
rule files at import time and therefore cannot run before they exist.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from translation_web_app.paths import LANGUAGE_RULES_DIR, RULES_DIR  # noqa: E402
from translation_web_app.prompt_modules import (  # noqa: E402
    AUDIT_CHECKLIST_RULES,
    AUDIT_GRADE_CRITERIA,
    AUDIT_INTRO,
    BX_STYLE_RULES,
    COMMON_LOCALIZATION_STANDARD,
    GLOSSARY_BRACKET_WRAP_RULE,
    GLOSSARY_DISCLAIMER_NAV_EXCEPTION,
    GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE,
    GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_EAST_ASIAN,
    GLOSSARY_NO_BRACKET_INSTRUCTION,
    GLOSSARY_TERM_RULES,
    LANGUAGE_LOCALIZATION_RULES,
    LANGUAGE_RULE_LABELS,
    TYPOGRAPHY_AND_PUNCTUATION_RULES,
)

SCHEMA_VERSION = 1
BODY_NOTICE = (
    "> The YAML front matter above is the only normative content. This body is\n"
    "> documentation for humans and agents and is never parsed by the app."
)


class FlowList(list):
    """A list rendered inline (``[a, b]``) instead of as a block sequence."""


yaml.add_representer(
    FlowList,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)


def slugify(canonical_key: str) -> str:
    return re.sub(r"[\s_]+", "-", canonical_key).lower()


def dump_front_matter(payload: dict) -> str:
    # width is effectively disabled: folded lines round-trip fine but produce
    # unreadable diffs on files whose whole point is being edited by hand.
    return yaml.dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        width=10**6,
        default_flow_style=False,
    )


def build_language_payload(canonical_key: str) -> dict:
    slug = slugify(canonical_key)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "language",
        "canonical_key": canonical_key,
        "display_name": LANGUAGE_RULE_LABELS[canonical_key],
        "rule_order": "sequence",
        "rules": [
            {
                "rule_id": f"{slug}-{index:03d}",
                "scope": FlowList(["app_prompt"]),
                "text": text,
            }
            for index, text in enumerate(LANGUAGE_LOCALIZATION_RULES[canonical_key], 1)
        ],
    }


def build_language_document(canonical_key: str) -> str:
    payload = build_language_payload(canonical_key)
    return (
        f"---\n{dump_front_matter(payload)}---\n\n"
        f"# {canonical_key}\n\n"
        f"{BODY_NOTICE}\n\n"
        f"Display name `{payload['display_name']}` is the heading shown above these\n"
        f"rules in the translation and audit prompts.\n\n"
        f"## Notes\n\n"
        f"(Rationale, source references, and open questions go here.)\n"
    )


def build_bx_payload() -> dict:
    rules: list[dict] = []
    for group, data in BX_STYLE_RULES["voice_attributes"].items():
        for index, text in enumerate(data["actionable_rules"], 1):
            rules.append(
                {
                    "rule_id": f"bx-{group.lower()}-{index:03d}",
                    "group": group,
                    "scope": FlowList(["app_prompt"]),
                    "text": text,
                }
            )
    for index, text in enumerate(BX_STYLE_RULES["negative_constraints"], 1):
        rules.append(
            {
                "rule_id": f"bx-negative-{index:03d}",
                "group": "NEGATIVE",
                "scope": FlowList(["app_prompt"]),
                "text": text,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "bx_style",
        "canonical_key": "bx_style",
        "display_name": "Samsung BX Style",
        "rule_order": "sequence",
        "identity": dict(BX_STYLE_RULES["system_identity"]),
        "rules": rules,
        "examples": [dict(item) for item in BX_STYLE_RULES["few_shot_examples"]],
    }


def build_bx_document() -> str:
    return (
        f"---\n{dump_front_matter(build_bx_payload())}---\n\n"
        f"# Samsung BX Style\n\n"
        f"{BODY_NOTICE}\n\n"
        f"`group` partitions the rules: OPEN / BOLD / AUTHENTIC become voice\n"
        f"attributes in the prompt, NEGATIVE becomes the constraint list. Group\n"
        f"order here is the order the model sees.\n\n"
        f"## Notes\n\n"
        f"(Rationale, source references, and open questions go here.)\n"
    )


def verify_language(canonical_key: str, document: str) -> None:
    """Reload the emitted YAML and require it to equal the source constant."""
    payload = yaml.safe_load(document.split("---\n", 2)[1])
    texts = [rule["text"] for rule in payload["rules"]]
    if texts != LANGUAGE_LOCALIZATION_RULES[canonical_key]:
        raise SystemExit(f"round-trip mismatch for {canonical_key}: rule text differs")
    if payload["display_name"] != LANGUAGE_RULE_LABELS[canonical_key]:
        raise SystemExit(f"round-trip mismatch for {canonical_key}: display_name differs")
    if payload["canonical_key"] != canonical_key:
        raise SystemExit(f"round-trip mismatch for {canonical_key}: canonical_key differs")


def verify_bx(document: str) -> None:
    payload = yaml.safe_load(document.split("---\n", 2)[1])
    rebuilt_voice: dict[str, dict] = {}
    negatives: list[str] = []
    for rule in payload["rules"]:
        if rule["group"] == "NEGATIVE":
            negatives.append(rule["text"])
        else:
            rebuilt_voice.setdefault(rule["group"], {"actionable_rules": []})
            rebuilt_voice[rule["group"]]["actionable_rules"].append(rule["text"])
    rebuilt = {
        "system_identity": payload["identity"],
        "voice_attributes": rebuilt_voice,
        "negative_constraints": negatives,
        "few_shot_examples": payload["examples"],
    }
    if rebuilt != BX_STYLE_RULES:
        raise SystemExit("round-trip mismatch for bx_style: structure differs")
    if list(rebuilt_voice) != list(BX_STYLE_RULES["voice_attributes"]):
        raise SystemExit("round-trip mismatch for bx_style: voice attribute order differs")


# The bracket-precedence sentence lives as a literal inside prompt_builder rather
# than a constant, so it is transcribed here once and asserted against the source
# file by verify_doc() below.
BRACKET_PRECEDENCE = (
    "Bracket precedence (highest first): a term marked no-bracket / 대괄호 제외 is never "
    "bracketed; inside a navigation path no term is bracketed; otherwise apply the generic "
    "bracket rule below. Term-specific exceptions always override the generic rule."
)

DOC_BUILDERS: dict[str, tuple[str, list[dict]]] = {}


def _doc_rule(slug: str, index: int, slot: str, text: str, label: str | None = None) -> dict:
    rule: dict = {
        "rule_id": f"{slug}-{index:03d}",
        "slot": slot,
        "scope": FlowList(["app_prompt"]),
    }
    if label is not None:
        rule["label"] = label
    rule["text"] = text
    return rule


def build_doc_payloads() -> dict[str, dict]:
    common = [
        _doc_rule("common", i, "standard", text)
        for i, text in enumerate(COMMON_LOCALIZATION_STANDARD["rules"], 1)
    ]
    typography = [
        _doc_rule("typography", i, "rule", text)
        for i, text in enumerate(TYPOGRAPHY_AND_PUNCTUATION_RULES["rules"], 1)
    ]
    glossary = [
        _doc_rule("glossary", 1, "term_rule", GLOSSARY_TERM_RULES["rules"][0]),
        _doc_rule("glossary", 2, "bracket_precedence", BRACKET_PRECEDENCE),
        _doc_rule("glossary", 3, "bracket_wrap", GLOSSARY_BRACKET_WRAP_RULE),
        _doc_rule("glossary", 4, "nav_exception", GLOSSARY_DISCLAIMER_NAV_EXCEPTION),
        _doc_rule("glossary", 5, "no_bracket", GLOSSARY_NO_BRACKET_INSTRUCTION),
        _doc_rule("glossary", 6, "nav_quote_default", GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE),
        _doc_rule("glossary", 7, "nav_quote_east_asian",
                  GLOSSARY_DISCLAIMER_NAV_QUOTE_RULE_EAST_ASIAN),
    ]
    audit = [_doc_rule("audit", 1, "intro", AUDIT_INTRO)]
    for i, (category, description) in enumerate(AUDIT_CHECKLIST_RULES, 2):
        audit.append(_doc_rule("audit", i, "checklist", description, label=category))
    offset = len(audit) + 1
    for i, (grade, criteria) in enumerate(AUDIT_GRADE_CRITERIA.items(), offset):
        audit.append(_doc_rule("audit", i, "grade", criteria, label=grade))

    specs = {
        "common": (COMMON_LOCALIZATION_STANDARD["name"], common),
        "typography": (TYPOGRAPHY_AND_PUNCTUATION_RULES["name"], typography),
        "glossary": ("Glossary Rules", glossary),
        "audit": ("Audit Criteria", audit),
    }
    return {
        kind: {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "canonical_key": kind,
            "display_name": display_name,
            "rule_order": "sequence",
            "rules": rules,
        }
        for kind, (display_name, rules) in specs.items()
    }


DOC_BODY_NOTES = {
    "common": "Applies to every target language, before the language-specific section.",
    "typography": "`display_name` is emitted as the prompt section heading, so it is load-bearing.",
    "glossary": (
        "`bracket_wrap` keeps its `{open}`/`{close}` placeholders — Python substitutes the\n"
        "locale's brackets. `nav_quote_default` and `nav_quote_east_asian` are the two arms of\n"
        "one branch; Python decides which applies."
    ),
    "audit": (
        "`checklist` labels are the `evaluation[].category` values the model echoes back.\n"
        "`grade` labels are the grade enum; three decoders depend on them, so a test pins the set."
    ),
}


def build_doc_document(kind: str, payload: dict) -> str:
    return (
        f"---\n{dump_front_matter(payload)}---\n\n"
        f"# {payload['display_name']}\n\n"
        f"{BODY_NOTICE}\n\n"
        f"{DOC_BODY_NOTES[kind]}\n\n"
        f"## Notes\n\n"
        f"(Rationale, source references, and open questions go here.)\n"
    )


def verify_doc(kind: str, payload: dict, document: str) -> None:
    """Reload the emitted YAML and require it to equal the source constants."""
    reloaded = yaml.safe_load(document.split("---\n", 2)[1])
    if [r["text"] for r in reloaded["rules"]] != [r["text"] for r in payload["rules"]]:
        raise SystemExit(f"round-trip mismatch for {kind}: rule text differs")
    if [r.get("slot") for r in reloaded["rules"]] != [r["slot"] for r in payload["rules"]]:
        raise SystemExit(f"round-trip mismatch for {kind}: slots differ")
    if [r.get("label") for r in reloaded["rules"]] != [r.get("label") for r in payload["rules"]]:
        raise SystemExit(f"round-trip mismatch for {kind}: labels differ")
    if kind == "glossary":
        # This sentence had no constant; it was a literal inside prompt_builder.
        # While that literal still existed we cross-checked the transcription
        # against it. Once prompt_builder reads the slot from Markdown the
        # literal is gone, and rules/glossary.md becomes the source — so the
        # check applies only when there is still something to compare against.
        source = (ROOT / "src/translation_web_app/prompt_builder.py").read_text(encoding="utf-8")

        def _flat(text: str) -> str:
            # Drop quote chars so Python's implicit concatenation across lines
            # ("...a " "b...") collapses to the runtime value before comparing.
            return " ".join(text.replace('"', "").split())

        if "Bracket precedence" in source and _flat(BRACKET_PRECEDENCE) not in _flat(source):
            raise SystemExit(
                "BRACKET_PRECEDENCE no longer matches the literal in prompt_builder.py"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(RULES_DIR))
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare generated output against disk without writing; exit 1 on mismatch",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    languages_dir = (
        LANGUAGE_RULES_DIR if out_dir == RULES_DIR else out_dir / "languages"
    )

    documents: dict[Path, str] = {out_dir / "bx_style.md": build_bx_document()}
    verify_bx(documents[out_dir / "bx_style.md"])
    for canonical_key in LANGUAGE_LOCALIZATION_RULES:
        document = build_language_document(canonical_key)
        verify_language(canonical_key, document)
        documents[languages_dir / f"{canonical_key}.md"] = document
    for kind, payload in build_doc_payloads().items():
        document = build_doc_document(kind, payload)
        verify_doc(kind, payload, document)
        documents[out_dir / f"{kind}.md"] = document

    if args.check:
        mismatches = [
            path
            for path, document in documents.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != document
        ]
        if mismatches:
            print(f"{len(mismatches)} file(s) differ from a fresh generation:")
            for path in sorted(mismatches):
                print(f"  - {path}")
            return 1
        print(f"{len(documents)} file(s) match a fresh generation.")
        return 0

    languages_dir.mkdir(parents=True, exist_ok=True)
    for path, document in sorted(documents.items()):
        path.write_text(document, encoding="utf-8", newline="\n")
    print(f"wrote {len(documents)} file(s) under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
