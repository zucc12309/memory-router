"""Filesystem operations for an Obsidian vault.

ObsidianVault owns all I/O. It guarantees:
  * writes stay inside the vault (path-traversal protection via safe_join),
  * existing notes are backed up before overwrite,
  * directory scaffolding and the privacy README/.gitignore exist.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

from .models import (
    GITIGNORE,
    INDEX_NOTE,
    README_NOTE,
    VAULT_FOLDERS,
)
from .utils import safe_join

_README_TEXT = """# Memory Router Vault

This vault is an **auto-generated projection** of your Memory Router database.
The source of truth is the local SQLite store at `~/.memory-router/` — this
vault is rebuilt from it and is safe to delete.

> [!warning] This vault may contain sensitive memories.
> Do **not** publish, commit, or cloud-sync this vault without reviewing its
> contents. Even with redaction enabled, treat it as private data.

## How it works

- Knowledge notes (Projects / Research / Decisions / People / Conversations)
  are consolidated summaries with `[[wikilinks]]` you can browse in Graph View.
- `90_Raw_Memories/` holds one note per memory (only when raw export is on).
- Mycelium associative edges become wikilinks; edge weights are kept in
  HTML comments like `[[CatBoost]] <!-- weight 0.82 -->`.

Regenerate any time with:

```bash
memory-router memory obsidian export
```
"""

_GITIGNORE_TEXT = """# Memory Router vault — keep private. Ignore everything.
*
"""

_INDEX_HEADER = """---
type: index
source: memory-router
---

# Memory Router Index

Auto-generated map of this vault. Open Graph View to explore associations.
"""


class ObsidianVault:
    """A safe, scaffolded Obsidian vault directory."""

    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser()

    # -- lifecycle ----------------------------------------------------------

    def init(self) -> "ObsidianVault":
        """Create the folder tree, README, .gitignore, and index. Idempotent."""
        self.path.mkdir(parents=True, exist_ok=True)
        for folder in VAULT_FOLDERS:
            (self.path / folder).mkdir(parents=True, exist_ok=True)
        # Privacy scaffolding — never overwrite a user-edited README.
        if not (self.path / README_NOTE).exists():
            self._raw_write(README_NOTE, _README_TEXT)
        if not (self.path / GITIGNORE).exists():
            self._raw_write(GITIGNORE, _GITIGNORE_TEXT)
        if not (self.path / INDEX_NOTE).exists():
            self._raw_write(INDEX_NOTE, _INDEX_HEADER)
        return self

    def is_initialized(self) -> bool:
        return self.path.exists() and (self.path / README_NOTE).exists()

    # -- note I/O -----------------------------------------------------------

    def exists(self, rel_path: str) -> bool:
        try:
            return safe_join(self.path, rel_path).exists()
        except ValueError:
            return False

    def read_note(self, rel_path: str) -> Optional[str]:
        target = safe_join(self.path, rel_path)
        if not target.exists():
            return None
        return target.read_text(encoding="utf-8")

    def write_note(self, rel_path: str, content: str, *, backup: bool = True) -> Path:
        """Write a note, backing up any existing file first.

        Parent folders are created on demand. Returns the absolute path.
        """
        target = safe_join(self.path, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if backup and target.exists():
            self.backup(rel_path)
        # Atomic-ish: write to a temp sibling then replace.
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(target)
        return target

    def backup(self, rel_path: str) -> Optional[Path]:
        """Copy an existing note into 99_Archive/.backups/ with a timestamp."""
        target = safe_join(self.path, rel_path)
        if not target.exists():
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = self.path / "99_Archive" / ".backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        dest = backup_dir / f"{stamp}__{target.name}"
        shutil.copy2(target, dest)
        return dest

    # -- internal -----------------------------------------------------------

    def _raw_write(self, rel_path: str, content: str) -> None:
        target = safe_join(self.path, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
