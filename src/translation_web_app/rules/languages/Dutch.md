---
schema_version: 1
kind: language
canonical_key: Dutch
display_name: Dutch Directness & Phrasing
rule_order: sequence
rules:
- rule_id: dutch-001
  scope: [app_prompt]
  text: Use 'u/uw' (formal address) consistently; do not use je/jij.
- rule_id: dutch-002
  scope: [app_prompt]
  text: Keep Dutch copy direct and concise.
- rule_id: dutch-003
  scope: [app_prompt]
  text: Avoid literal translation of English word order or noun-phrase structures.
- rule_id: dutch-004
  scope: [app_prompt]
  text: Use typographic apostrophes (’) where Dutch requires an apostrophe in user-facing copy. Do not mix them with straight ASCII apostrophes within the same story; preserve official names and exact technical strings as styled.
---

# Dutch

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `Dutch Directness & Phrasing` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
