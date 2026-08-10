---
schema_version: 1
kind: language
canonical_key: Japanese
display_name: Japanese ます-form Consistency
rule_order: sequence
rules:
- rule_id: japanese-001
  scope: [app_prompt]
  text: Use consistent ます-form unless project guidance specifies otherwise.
- rule_id: japanese-002
  scope: [app_prompt]
  text: Use natural Japanese UI phrasing and avoid close structural calques from English or Korean.
- rule_id: japanese-003
  scope: [app_prompt]
  text: Prioritize natural '操作/設定' style expressions over mechanical literal translations.
- rule_id: japanese-004
  scope: [app_prompt]
  text: Avoid excessive honorifics unless the context clearly requires them.
---

# Japanese

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `Japanese ます-form Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
