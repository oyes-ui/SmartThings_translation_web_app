---
schema_version: 1
kind: language
canonical_key: Turkish
display_name: Turkish UI Phrasing Consistency
rule_order: sequence
rules:
- rule_id: turkish-001
  scope: [app_prompt]
  text: Use natural Turkish word order and avoid structural calques from English.
- rule_id: turkish-002
  scope: [app_prompt]
  text: Maintain concise imperative or descriptive forms suitable for UI copy.
- rule_id: turkish-003
  scope: [app_prompt]
  text: Use typographic apostrophes (’) when separating suffixes from proper names, abbreviations, and numerals, and in other orthographically required uses. Do not mix them with straight ASCII apostrophes within the same story; preserve official names and exact technical strings as styled.
---

# Turkish

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `Turkish UI Phrasing Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
