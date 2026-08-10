---
schema_version: 1
kind: language
canonical_key: Russian
display_name: Russian Word Order & Phrasing
rule_order: sequence
rules:
- rule_id: russian-001
  scope: [app_prompt]
  text: Use natural Russian word order.
- rule_id: russian-002
  scope: [app_prompt]
  text: Avoid English-style noun-phrase literal translations and excessive capitalization.
- rule_id: russian-003
  scope: [app_prompt]
  text: Use «...» (guillemets) for quoted text and navigation paths; do not use English straight quotation marks.
---

# Russian

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `Russian Word Order & Phrasing` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
