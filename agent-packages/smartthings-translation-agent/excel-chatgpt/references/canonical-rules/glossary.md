---
schema_version: 1
kind: glossary
canonical_key: glossary
display_name: Glossary Rules
rule_order: sequence
rules:
- rule_id: glossary-001
  slot: term_rule
  scope: [app_prompt]
  text: Use provided glossary terms exactly as given — including capitalization, spacing, and market variants. Glossary capitalization is authoritative and overrides title case, sentence case, and heading/button capitalization rules; do not adapt glossary terms for naturalness.
- rule_id: glossary-002
  slot: bracket_precedence
  scope: [app_prompt]
  text: 'Bracket precedence (highest first): a term marked no-bracket / 대괄호 제외 is never bracketed; inside a navigation path no term is bracketed; otherwise apply the generic bracket rule below. Term-specific exceptions always override the generic rule.'
- rule_id: glossary-003
  slot: bracket_wrap
  scope: [app_prompt]
  text: Wrap glossary terms in '{open}' and '{close}'.
- rule_id: glossary-004
  slot: nav_exception
  scope: [app_prompt]
  text: 'Exception: do not wrap terms that appear inside a navigation path (e.g., Settings > Device). This exception is per-occurrence: if the same glossary term also appears outside the navigation path, still wrap that outside occurrence — only the occurrence inside the path stays unwrapped.'
- rule_id: glossary-005
  slot: no_bracket
  scope: [app_prompt]
  text: For titles, section headings, and buttons, render standalone glossary terms as plain text. Do not add quotation marks, guillemets, brackets, or any other wrapper around them ([], 「」, «», "", “”, etc.), even if the source text contains such marks. Use quotation marks only for an explicitly quoted UI navigation path or source text that requires quotation. Do not change glossary capitalization to satisfy heading or button case style.
- rule_id: glossary-006
  slot: nav_quote_default
  scope: [app_prompt]
  text: Enclose navigation paths in quotation marks appropriate for the target language. Place the sentence-ending period outside the closing quotation mark.
- rule_id: glossary-007
  slot: nav_quote_east_asian
  scope: [app_prompt]
  text: Enclose navigation paths in quotation marks appropriate for the target language.
- rule_id: glossary-008
  slot: disclaimer_linebreak
  scope: [app_prompt]
  text: Disclaimer text always starts with '* ' (an asterisk followed by a space). If the disclaimer spans multiple lines, every line — not just the first — must start with '* '.
---

# Glossary Rules

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

`bracket_wrap` keeps its `{open}`/`{close}` placeholders — Python substitutes the
locale's brackets. `nav_quote_default` and `nav_quote_east_asian` are the two arms of
one branch; Python decides which applies.

## Notes

(Rationale, source references, and open questions go here.)
