---
schema_version: 1
kind: language
canonical_key: Traditional Chinese
display_name: Traditional Chinese Consistency
rule_order: sequence
rules:
- rule_id: traditional-chinese-001
  scope: [app_prompt]
  text: Use natural Taiwan Traditional Chinese wording and characters.
- rule_id: traditional-chinese-002
  scope: [app_prompt]
  text: Avoid Mainland Chinese-specific terminology.
- rule_id: traditional-chinese-003
  scope: [app_prompt]
  text: Use fullwidth punctuation marks (，。！？) consistently; avoid half-width ASCII punctuation.
- rule_id: traditional-chinese-004
  scope: [app_prompt]
  text: Use 「...」 for quoted text and navigation paths.
---

# Traditional Chinese

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `Traditional Chinese Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
