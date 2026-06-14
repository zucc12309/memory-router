"""Tests for ObsidianVault: scaffolding, safe writes, backups, traversal."""

from __future__ import annotations

import pytest

from memory_router.memory.obsidian.models import VAULT_FOLDERS
from memory_router.memory.obsidian.vault import ObsidianVault


class TestInit:
    def test_creates_folder_tree(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        for folder in VAULT_FOLDERS:
            assert (vault.path / folder).is_dir()

    def test_creates_readme_and_gitignore(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        assert (vault.path / "README.md").exists()
        assert (vault.path / ".gitignore").exists()
        assert (vault.path / "Memory Router Index.md").exists()

    def test_gitignore_ignores_everything(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        assert (vault.path / ".gitignore").read_text().strip().endswith("*")

    def test_readme_has_privacy_warning(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        text = (vault.path / "README.md").read_text()
        assert "sensitive" in text.lower()

    def test_idempotent(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v")
        vault.init()
        # Edit the README, re-init must not clobber user edits.
        readme = vault.path / "README.md"
        readme.write_text("MY EDITS")
        vault.init()
        assert readme.read_text() == "MY EDITS"

    def test_is_initialized(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v")
        assert not vault.is_initialized()
        vault.init()
        assert vault.is_initialized()


class TestNoteIO:
    def test_write_and_read(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        vault.write_note("01_Projects/note.md", "hello")
        assert vault.read_note("01_Projects/note.md") == "hello"

    def test_read_missing_returns_none(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        assert vault.read_note("nope.md") is None

    def test_exists(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        assert not vault.exists("01_Projects/x.md")
        vault.write_note("01_Projects/x.md", "data")
        assert vault.exists("01_Projects/x.md")

    def test_creates_parent_dirs(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        vault.write_note("deep/nested/path/note.md", "x")
        assert vault.read_note("deep/nested/path/note.md") == "x"


class TestBackup:
    def test_overwrite_creates_backup(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        vault.write_note("01_Projects/n.md", "v1")
        vault.write_note("01_Projects/n.md", "v2")
        backups = list((vault.path / "99_Archive" / ".backups").glob("*__n.md"))
        assert len(backups) == 1
        assert backups[0].read_text() == "v1"
        assert vault.read_note("01_Projects/n.md") == "v2"

    def test_backup_missing_returns_none(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        assert vault.backup("does/not/exist.md") is None

    def test_first_write_no_backup(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        vault.write_note("01_Projects/fresh.md", "data")
        backup_dir = vault.path / "99_Archive" / ".backups"
        assert not backup_dir.exists() or not list(backup_dir.glob("*fresh.md"))


class TestPathTraversal:
    def test_write_escape_rejected(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        with pytest.raises(ValueError):
            vault.write_note("../../evil.md", "pwned")

    def test_exists_escape_is_false(self, tmp_path):
        vault = ObsidianVault(tmp_path / "v").init()
        assert vault.exists("../../../etc/passwd") is False
