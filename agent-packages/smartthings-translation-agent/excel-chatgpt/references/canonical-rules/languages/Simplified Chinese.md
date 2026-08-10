---
schema_version: 1
kind: language
canonical_key: Simplified Chinese
display_name: Simplified Chinese Consistency
rule_order: sequence
rules:
- rule_id: simplified-chinese-001
  scope: [app_prompt]
  text: Use natural Mainland Chinese wording and Simplified Chinese characters.
- rule_id: simplified-chinese-002
  scope: [app_prompt]
  text: Avoid Taiwan-specific terminology.
- rule_id: simplified-chinese-003
  scope: [app_prompt]
  text: Use fullwidth punctuation marks (，。！？) consistently; avoid half-width ASCII punctuation.
- rule_id: simplified-chinese-004
  scope: [app_prompt]
  text: Use fullwidth double quotation marks (“...”) for quoted text and navigation paths; do not use 「」 (Japanese-style brackets).
- rule_id: simplified-chinese-005
  scope: [app_prompt]
  text: Use 您 rather than 你 in app/marketing copy to maintain a polished, respectful register.
- rule_id: simplified-chinese-006
  scope: [app_prompt]
  text: Avoid overly casual spoken-style (口语化) wording, chatty particles (e.g., 呀, 啦, 呢, 哦), and colloquial sentence flow.
- rule_id: simplified-chinese-007
  scope: [app_prompt]
  text: Use a polished Mainland Chinese register suited to premium tech and marketing copy. Recompose rather than translate word-for-word; avoid bureaucratic wording and structural translationese (e.g., …的情况下 or unnecessary 并), using clear, idiomatic benefit-led wording where appropriate (e.g., 无需…即可), without forcing a fixed sentence pattern, character count, or vocabulary choice.
- rule_id: simplified-chinese-008
  scope: [app_prompt]
  text: Use precise technical wording for device actions and preserve source-confirmed qualifiers, including remote operation. State a specific UI location only when explicitly supported by the source or context; never invent one.
---

# Simplified Chinese

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Display name `Simplified Chinese Consistency` is the heading shown above these
rules in the translation and audit prompts.

## Notes

(Rationale, source references, and open questions go here.)
