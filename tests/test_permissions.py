from __future__ import annotations

import os
import stat

from memory_router import config as cfg_mod
from memory_router.memory.sqlite_store import MemoryStore


def _patch_root(monkeypatch, root):
    monkeypatch.setattr(cfg_mod, "ROOT_DIR", root)
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", root / "config.yaml")
    monkeypatch.setattr(cfg_mod, "CONVERSATIONS_DB", root / "conversations.sqlite")
    monkeypatch.setattr(cfg_mod, "MEMORIES_DB", root / "memories.sqlite")
    monkeypatch.setattr(cfg_mod, "VECTOR_DIR", root / "vector_index")
    monkeypatch.setattr(cfg_mod, "LOG_DIR", root / "logs")


def test_config_files_are_locked_down(tmp_path, monkeypatch):
    root = tmp_path / "memory-router"
    _patch_root(monkeypatch, root)

    cfg_mod.ensure_dirs()
    cfg_mod.save_config(cfg_mod.Config())

    assert stat.S_IMODE(os.stat(root).st_mode) & 0o077 == 0
    assert stat.S_IMODE(os.stat(cfg_mod.CONFIG_PATH).st_mode) & 0o077 == 0


def test_sqlite_files_are_locked_down(tmp_path, monkeypatch):
    root = tmp_path / "memory-router"
    _patch_root(monkeypatch, root)

    MemoryStore(path=root / "memories.sqlite")

    assert stat.S_IMODE(os.stat(root / "memories.sqlite").st_mode) & 0o077 == 0
