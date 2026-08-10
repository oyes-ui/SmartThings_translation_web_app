---
schema_version: 1
kind: language
canonical_key: French_Canada
display_name: Canadian French Consistency
rule_order: sequence
rules:
- rule_id: french-canada-001
  scope: [app_prompt]
  text: Use Vous-form consistently; do not use Tu-form.
- rule_id: french-canada-002
  scope: [app_prompt]
  text: Use «...» (guillemets) for quoted text and navigation paths.
- rule_id: french-canada-003
  scope: [app_prompt]
  text: Follow Canadian French standards; prioritize phrasing natural to North American French over mainland France idioms.
---

# French_Canada

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `Canadian French Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
