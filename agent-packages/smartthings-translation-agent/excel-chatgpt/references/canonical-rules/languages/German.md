---
schema_version: 1
kind: language
canonical_key: German
display_name: German Du-form Consistency
rule_order: sequence
rules:
- rule_id: german-001
  scope: [app_prompt]
  text: Use Du-form consistently unless the locale or project explicitly requires Sie-form.
- rule_id: german-002
  scope: [app_prompt]
  text: Ensure natural capitalization of nouns and maintain natural compound word structures.
- rule_id: german-003
  scope: [app_prompt]
  text: Avoid overly formal or technical wording in short UI copy.
- rule_id: german-004
  scope: [app_prompt]
  text: Use „...“ (German quotation marks) for quoted text and navigation paths; do not use English straight quotation marks.
---

# German

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `German Du-form Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
