---
schema_version: 1
kind: typography
canonical_key: typography
display_name: Typography and Punctuation Rules
rule_order: sequence
rules:
- rule_id: typography-001
  slot: rule
  scope: [app_prompt]
  text: For quotation marks, spacing, and punctuation specific to the target language, follow the rules stated in the language section above; where no specific rule is given, apply the standard convention for that language and locale.
- rule_id: typography-002
  slot: rule
  scope: [app_prompt]
  text: Do not mechanically copy English punctuation, quotation mark placement, spacing, or sentence-ending style into other languages.
---

# Typography and Punctuation Rules

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

`display_name` is emitted as the prompt section heading, so it is load-bearing.

## Notes

(Rationale, source references, and open questions go here.)
