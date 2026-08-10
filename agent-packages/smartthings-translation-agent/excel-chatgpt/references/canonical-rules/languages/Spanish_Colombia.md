---
schema_version: 1
kind: language
canonical_key: Spanish_Colombia
display_name: Colombian Spanish — Tú Form
locale: es_CO
rule_order: sequence
rules:
- rule_id: spanish-colombia-001
  scope: [app_prompt, agent_audit]
  text: Use tú (informal address) consistently; do not use usted unless the source or project explicitly requires formal register.
- rule_id: spanish-colombia-002
  scope: [app_prompt]
  text: Do not use vos or voseo verb forms unless the market or campaign brief explicitly specifies them.
- rule_id: spanish-colombia-003
  scope: [app_prompt]
  text: Use neutral Latin American vocabulary (e.g., 'celular', 'computador'); avoid Spain-specific wording (e.g., 'móvil', 'ordenador', 'vosotros') and strongly regional colloquialisms (e.g., 'ratico', 'momentico').
- rule_id: spanish-colombia-004
  scope: [app_prompt]
  text: Address plural users with ustedes and third-person plural verb forms; never use vosotros.
- rule_id: spanish-colombia-005
  scope: [agent_audit]
  severity: major
  text: Verify tú consistency across verb conjugation, the possessive tu, the object pronoun te, and imperatives (Configura / Revisa / Puedes, never Configure / Revise / Puede). Exclude short subject-less noun-form UI labels such as "Configuración", where no conjugation is present to check.
- rule_id: spanish-colombia-006
  scope: [agent_audit]
  severity: major
  text: Flag any vos or voseo usage and confirm whether the market or campaign brief explicitly allows it before accepting.
---

# Spanish_Colombia

> The YAML front matter above is the only normative content. This body is
> documentation for humans and agents and is never parsed by the app.

Sheet code `CO(콜롬비아)` · locale `es_CO` · US source group.
Kept separate from `Spanish` (Spain / Castilian), which is unchanged.

## Why tú is the default

Colombia is not tú-uniform: usted is common in Medellín and much of Antioquia even
between intimates, vos is live in several regions, and Bogotá skews tú. So `tú` here
is **a deliberate SmartThings CO brand-voice default, not "standard Colombian
Spanish"**. It follows the same pattern already used for `Spanish` and `German`:
informal by default, formal only when a brief says so.

Legal notices, support copy, and account/security/billing strings are covered by the
same default — usted applies only when the project brief specifies formal register.
There is no separate metadata field for this; it is a prompt-text rule, matching how
Spain Spanish and German already express the same exception.

## Why the conjugation check is agent-only

`spanish-colombia-005` and `-006` are `scope: [agent_audit]`, so they never enter the
app's translation or audit prompt — they are reachable only through
`rules_loader.get_rules().rules_for_scope("agent_audit")`. Consequence worth knowing:
the app's own LLM auditor does not check tú-conjugation consistency for CO; only the
agent does.

They are agent-only because they require reading a whole cell and deciding whether a
form is checkable at all. Short noun-form labels ("Configuración", "Ajustes") carry no
conjugation, so a blanket prompt instruction to "keep tú conjugation consistent" would
invite the model to invent a verb where the source has none.

`severity: major` on both was assigned during the §2 review, not derived from data.

## RAG

`rag_retriever` matches `target_lang` exactly, so `es_CO` / `CO(콜롬비아)` rows are
isolated from `ES(스페인)` with no code change. Until CO cases accumulate, prefer CO
results and treat general Spanish cases as lower-priority reference only — Spain
vocabulary re-entering through RAG is the main drift risk for this locale.

## Notes

Out of scope for this locale: date, number, currency (COP), and address formatting.
This app edits and audits text content; it does not render those.
