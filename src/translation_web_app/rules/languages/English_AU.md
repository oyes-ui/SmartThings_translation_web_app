---
schema_version: 1
kind: language
canonical_key: English_AU
display_name: Australian English Consistency
rule_order: sequence
rules:
- rule_id: english-au-001
  scope: [app_prompt]
  text: Use Australian English with British-style spelling.
- rule_id: english-au-002
  scope: [app_prompt]
  text: Avoid overly aggressive US-style marketing tones; keep it helpful and clear.
- rule_id: english-au-003
  scope: [app_prompt]
  text: Use straight ASCII apostrophes (') for contractions and possessives in user-facing copy. Do not mix them with typographic apostrophes within the same story; preserve official names and exact technical strings as styled.
---

# English_AU

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `Australian English Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
