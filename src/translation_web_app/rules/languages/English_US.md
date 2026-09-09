---
schema_version: 1
kind: language
canonical_key: English_US
display_name: US English Consistency
rule_order: sequence
rules:
- rule_id: english-us-001
  scope: [app_prompt]
  text: Use US English spelling and wording (e.g., 'color', 'personalize').
- rule_id: english-us-002
  scope: [app_prompt]
  text: 'Follow the project-specific rule for disclaimers: place the sentence-ending period outside the closing quotation mark.'
- rule_id: english-us-003
  scope: [app_prompt]
  text: Use straight ASCII apostrophes (') for contractions and possessives in user-facing copy. Do not mix them with typographic apostrophes within the same story; preserve official names and exact technical strings as styled.
---

# English_US

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `US English Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
