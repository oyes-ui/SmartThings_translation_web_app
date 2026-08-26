# AGENTS.md

This folder is the SmartThings Translation Agent skill package.

- Use `SKILL.md` as the primary instruction source, then read the `references/` document for the
  task at hand before acting (see the routing table at the top of `SKILL.md`).
- Treat this folder as the execution root for translation review, RAG, Excel, and rules work.
- Connect to the SmartThings app repo through `scripts/bootstrap.py --app-root <path>`.
- Before any `.xlsx` write, read the mandatory “0단계” gates in `references/excel-workflow.md`.
  Declare separate value, structure, and formatting authorities; custom one-off scripts are not exempt.
  New one-off writers must import `scripts/workbook_mutation_guard.py`; extend that shared module with tests
  instead of copying or privately reimplementing its safety primitives.
  Do not promote a workbook until an independent read-back verifier passes all declared invariants.
- App orchestration maintenance files are out of scope unless the user explicitly asks for that work.
