---
schema_version: 1
kind: language
canonical_key: Italian
display_name: Italian UI Phrasing Consistency
rule_order: sequence
rules:
- rule_id: italian-001
  scope: [app_prompt]
  text: Use natural Italian UI sentence structures; avoid English-style noun-chaining.
- rule_id: italian-002
  scope: [app_prompt]
  text: Use typographic apostrophes (’) for elision and other apostrophe uses in user-facing copy. Do not mix them with straight ASCII apostrophes within the same story; preserve official names and exact technical strings as styled.
---

# Italian

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `Italian UI Phrasing Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
