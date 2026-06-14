"""CLI tests for `memory-router memory obsidian` using an isolated HOME.

The CLI binds storage paths from config at import time, so we point HOME at a
tmp dir and reload config + sqlite_store + cli (mirrors test_mcp_server_large_prompt).
"""

from __future__ import annotations

import importlib

from typer.testing import CliRunner

runner = CliRunner()


def _fresh_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    import memory_router.config as config_mod
    import memory_router.memory.sqlite_store as store_mod
    import memory_router.cli as cli_mod

    config_mod = importlib.reload(config_mod)
    store_mod = importlib.reload(store_mod)
    cli_mod = importlib.reload(cli_mod)
    return cli_mod, config_mod, store_mod


def _init_app(cli_mod, config_mod):
    """Write a minimal config so _require_init() passes."""
    config_mod.ensure_dirs()
    config_mod.save_config(config_mod.Config())


class TestObsidianInit:
    def test_init_creates_vault_and_enables(self, monkeypatch, tmp_path):
        cli_mod, config_mod, _ = _fresh_cli(monkeypatch, tmp_path)
        _init_app(cli_mod, config_mod)
        vault_dir = tmp_path / "MyVault"

        result = runner.invoke(
            cli_mod.app, ["memory", "obsidian", "init", "--vault", str(vault_dir)]
        )
        assert result.exit_code == 0, result.output
        assert (vault_dir / "README.md").exists()
        assert (vault_dir / "01_Projects").is_dir()

        cfg = config_mod.load_config()
        assert cfg.obsidian_enabled is True
        assert cfg.obsidian_vault_path == str(vault_dir.resolve())


class TestObsidianExport:
    def test_export_requires_enabled(self, monkeypatch, tmp_path):
        cli_mod, config_mod, _ = _fresh_cli(monkeypatch, tmp_path)
        _init_app(cli_mod, config_mod)  # obsidian disabled by default
        result = runner.invoke(cli_mod.app, ["memory", "obsidian", "export"])
        assert result.exit_code == 1
        assert "disabled" in result.output.lower()

    def test_full_flow_init_export_status(self, monkeypatch, tmp_path):
        cli_mod, config_mod, store_mod = _fresh_cli(monkeypatch, tmp_path)
        _init_app(cli_mod, config_mod)

        # Seed a memory.
        store = store_mod.MemoryStore()
        store.add(store_mod.Memory(
            content="RideCompare uses CatBoost", domain="software",
            task="explain", concepts=["RideCompare", "CatBoost"]))

        vault_dir = tmp_path / "Vault"
        assert runner.invoke(
            cli_mod.app, ["memory", "obsidian", "init", "--vault", str(vault_dir)]
        ).exit_code == 0

        export = runner.invoke(cli_mod.app, ["memory", "obsidian", "export", "--all"])
        assert export.exit_code == 0, export.output
        assert "Obsidian export" in export.output
        assert list((vault_dir / "90_Raw_Memories").glob("*.md"))

        status = runner.invoke(cli_mod.app, ["memory", "obsidian", "status"])
        assert status.exit_code == 0
        assert "Vault initialized" in status.output
        assert "Notes exported" in status.output

    def test_export_project(self, monkeypatch, tmp_path):
        cli_mod, config_mod, store_mod = _fresh_cli(monkeypatch, tmp_path)
        _init_app(cli_mod, config_mod)
        store = store_mod.MemoryStore()
        store.add(store_mod.Memory(
            content="RideCompare detail", domain="software",
            task="explain", concepts=["RideCompare"]))
        vault_dir = tmp_path / "Vault"
        runner.invoke(cli_mod.app, ["memory", "obsidian", "init", "--vault", str(vault_dir)])
        result = runner.invoke(
            cli_mod.app, ["memory", "obsidian", "export", "--project", "RideCompare"]
        )
        assert result.exit_code == 0
        assert "Knowledge notes" in result.output


class TestObsidianStatus:
    def test_status_disabled_by_default(self, monkeypatch, tmp_path):
        cli_mod, config_mod, _ = _fresh_cli(monkeypatch, tmp_path)
        _init_app(cli_mod, config_mod)
        result = runner.invoke(cli_mod.app, ["memory", "obsidian", "status"])
        assert result.exit_code == 0
        assert "Obsidian status" in result.output
        assert "no" in result.output.lower()  # enabled = no
