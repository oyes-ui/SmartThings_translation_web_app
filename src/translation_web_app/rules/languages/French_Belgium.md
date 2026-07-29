---
schema_version: 1
kind: language
canonical_key: French_Belgium
display_name: Belgian French Consistency
rule_order: sequence
rules:
- rule_id: french-belgium-001
  scope: [app_prompt]
  text: Use Vous-form consistently; do not use Tu-form.
- rule_id: french-belgium-002
  scope: [app_prompt]
  text: Use «...» (guillemets) for quoted text and navigation paths.
- rule_id: french-belgium-003
  scope: [app_prompt]
  text: Use neutral French and avoid overly idiomatic expressions specific to mainland France.
- rule_id: french-belgium-004
  scope: [app_prompt]
  text: Ensure consistent tone for the Belgian market.
---

# French_Belgium

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `Belgian French Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
