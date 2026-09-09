---
schema_version: 1
kind: language
canonical_key: French
display_name: French Tone and Consistency
rule_order: sequence
rules:
- rule_id: french-001
  scope: [app_prompt]
  text: Use Vous-form consistently; do not use Tu-form.
- rule_id: french-002
  scope: [app_prompt]
  text: Use «...» (guillemets) for quoted text and navigation paths.
- rule_id: french-003
  scope: [app_prompt]
  text: Avoid unnecessary capitalization in UI copy.
- rule_id: french-004
  scope: [app_prompt]
  text: Use natural French phrasing and avoid English-influenced structures.
- rule_id: french-005
  scope: [app_prompt]
  text: Use typographic apostrophes (’) for elision and other apostrophe uses in user-facing copy. Do not mix them with straight ASCII apostrophes within the same story; preserve official names and exact technical strings as styled.
---

# French

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `French Tone and Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
