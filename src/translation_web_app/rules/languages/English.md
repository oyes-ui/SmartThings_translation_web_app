---
schema_version: 1
kind: language
canonical_key: English
display_name: US English Consistency
rule_order: sequence
rules:
- rule_id: english-001
  scope: [app_prompt]
  text: Use US English spelling and wording (e.g., 'color', 'personalize') unless a more specific English market variant is specified.
- rule_id: english-002
  scope: [app_prompt]
  text: 'Follow the project-specific rule for disclaimers: place the sentence-ending period outside the closing quotation mark.'
---

# English

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `US English Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
