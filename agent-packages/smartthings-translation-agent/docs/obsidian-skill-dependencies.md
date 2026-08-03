# Obsidian Skill Dependencies

The SmartThings Translation Agent uses the following **optional global Codex skills** for vault integration. They are not vendored in this repository.

| Skill | Purpose |
| --- | --- |
| `obsidian-markdown` | Properties, wikilinks, callouts, and valid Obsidian Markdown |
| `obsidian-cli` | Search and inspect a running Obsidian vault |
| `obsidian-bases` | Read-only review status dashboard |

## Source and license

- Upstream: <https://github.com/kepano/obsidian-skills>
- License: MIT License, Copyright (c) 2026 Steph Ango (@kepano)
- Installed revision: `a1dc48e68138490d522c04cbf5822214c6eb1202` (upstream `main` at installation)

The project does not copy upstream skill files. If they are ever vendored, keep the upstream copyright notice and full MIT license with the copied files.

## Install or update

```bash
python3 ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo kepano/obsidian-skills \
  --path skills/obsidian-markdown skills/obsidian-cli skills/obsidian-bases
```

Installed skills become available to Codex on its next turn. Verify the local state with:

```bash
python scripts/obsidian_workflow.py status
```

If the skills or a running Obsidian CLI are unavailable, report staging and filesystem search still work; do not claim CLI-backed vault access.
