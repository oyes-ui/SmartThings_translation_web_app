---
schema_version: 1
kind: common
canonical_key: common
display_name: Common Localization Standard
rule_order: sequence
rules:
- rule_id: common-001
  slot: standard
  scope: [app_prompt]
  text: Preserve the original intent, nuance, and user benefit of the source.
- rule_id: common-002
  slot: standard
  scope: [app_prompt]
  text: Avoid overly literal translation when more natural, market-appropriate wording communicates the same intent better.
- rule_id: common-003
  slot: standard
  scope: [app_prompt]
  text: Where appropriate, use creative expression — refined wit, metaphor, or personification — to make technology feel approachable rather than technical.
- rule_id: common-004
  slot: standard
  scope: [app_prompt]
  text: Avoid culturally awkward idioms, fear-based framing, overly formal or technical language, and hedging words (e.g., 'hopefully', 'try to', 'might').
- rule_id: common-005
  slot: standard
  scope: [app_prompt]
  text: Keep UI copy concise while keeping the action or benefit clear.
---

# Common Localization Standard

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Applies to every target language, before the language-specific section.

## Notes

(Rationale, source references, and open questions go here.)
