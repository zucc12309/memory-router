"""Export manifest for idempotent, incremental Obsidian exports.

A small JSON file at the vault root records a content hash per note so the
exporter can skip notes that have not changed. This is what makes repeated
``export`` runs cheap and duplicate-free.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict

from .models import MANIFEST


def content_hash(content: str) -> str:
    """Stable SHA-256 of note content (sans volatile backup noise)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class VaultIndex:
    """Tracks exported notes by content hash for incremental updates."""

    def __init__(self, vault_path: Path | str):
        self.path = Path(vault_path).expanduser() / MANIFEST
        self._notes: Dict[str, str] = {}
        self.last_export: float = 0.0
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        self._notes = dict(data.get("notes", {}))
        self.last_export = float(data.get("last_export", 0.0))

    def save(self) -> None:
        self.last_export = time.time()
        payload = {
            "version": 1,
            "last_export": self.last_export,
            "notes": self._notes,
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- change detection ---------------------------------------------------

    def is_unchanged(self, rel_path: str, content: str) -> bool:
        """True if this exact content was already exported for rel_path."""
        return self._notes.get(rel_path) == content_hash(content)

    def record(self, rel_path: str, content: str) -> None:
        self._notes[rel_path] = content_hash(content)

    def forget(self, rel_path: str) -> None:
        self._notes.pop(rel_path, None)

    # -- introspection ------------------------------------------------------

    @property
    def note_count(self) -> int:
        return len(self._notes)

    def tracked_paths(self) -> list[str]:
        return sorted(self._notes.keys())
