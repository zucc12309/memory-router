"""Tests for configuration management."""

from pathlib import Path

from memory_router.config import Config, save_config, load_config, set_value, DEFAULT_MODELS


def test_config_defaults():
    cfg = Config()
    assert cfg.mode == "local"
    assert cfg.local_model == ""
    assert cfg.token_budget == 4000
    assert cfg.mycelium_enabled is True
    assert cfg.memory_decay_enabled is True
    assert cfg.adaptive_routing is False
    assert cfg.encryption_enabled is False
    assert cfg.mcp_rate_limit == 100


def test_config_roundtrip(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("memory_router.config.CONFIG_PATH", config_path)
    monkeypatch.setattr("memory_router.config.ROOT_DIR", tmp_path)

    cfg = Config(mode="hybrid", token_budget=8000, mycelium_enabled=False)
    cfg.local_model = "llama3.1:8b"
    save_config(cfg)

    loaded = load_config()
    assert loaded.mode == "hybrid"
    assert loaded.token_budget == 8000
    assert loaded.mycelium_enabled is False
    assert loaded.local_model == "llama3.1:8b"


def test_config_from_dict_backfills_models():
    cfg = Config.from_dict({"mode": "api", "models": {"openai_small": "gpt-4o-mini"}})
    assert cfg.mode == "api"
    # Should have backfilled missing model keys
    assert "anthropic_small" in cfg.models
    assert "gemini_small" in cfg.models
    # User override preserved
    assert cfg.models["openai_small"] == "gpt-4o-mini"


def test_config_from_dict_drops_invalid_local_model():
    cfg = Config.from_dict({"mode": "local", "local_model": "yes"})
    assert cfg.local_model == ""


def test_set_value_bool(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("memory_router.config.CONFIG_PATH", config_path)
    monkeypatch.setattr("memory_router.config.ROOT_DIR", tmp_path)

    save_config(Config())
    cfg = set_value("adaptive_routing", "true")
    assert cfg.adaptive_routing is True

    cfg = set_value("adaptive_routing", "false")
    assert cfg.adaptive_routing is False


def test_set_value_int(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("memory_router.config.CONFIG_PATH", config_path)
    monkeypatch.setattr("memory_router.config.ROOT_DIR", tmp_path)

    save_config(Config())
    cfg = set_value("token_budget", "8000")
    assert cfg.token_budget == 8000


def test_set_value_unknown_key(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("memory_router.config.CONFIG_PATH", config_path)
    monkeypatch.setattr("memory_router.config.ROOT_DIR", tmp_path)

    save_config(Config())
    try:
        set_value("nonexistent_key", "value")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_set_value_invalid_mode(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("memory_router.config.CONFIG_PATH", config_path)
    monkeypatch.setattr("memory_router.config.ROOT_DIR", tmp_path)
    save_config(Config())

    try:
        set_value("mode", "invalid_mode")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid mode" in str(e)


def test_set_value_invalid_provider(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("memory_router.config.CONFIG_PATH", config_path)
    monkeypatch.setattr("memory_router.config.ROOT_DIR", tmp_path)
    save_config(Config())

    try:
        set_value("default_provider", "fakeprovider")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid provider" in str(e)


def test_set_value_invalid_local_model(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("memory_router.config.CONFIG_PATH", config_path)
    monkeypatch.setattr("memory_router.config.ROOT_DIR", tmp_path)
    save_config(Config())

    try:
        set_value("local_model", "yes")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "local_model" in str(e)


def test_set_value_range_validation(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("memory_router.config.CONFIG_PATH", config_path)
    monkeypatch.setattr("memory_router.config.ROOT_DIR", tmp_path)
    save_config(Config())

    try:
        set_value("token_budget", "50")  # Below minimum of 100
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "between" in str(e)

    try:
        set_value("mcp_rate_limit", "0")  # Below minimum of 1
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "between" in str(e)


def test_default_models_complete():
    """All expected model tier keys should be present in defaults."""
    expected = [
        "local_simple", "local_default",
        "openai_small", "openai_large",
        "anthropic_small", "anthropic_mid", "anthropic_large",
        "gemini_small", "gemini_mid", "gemini_large",
    ]
    for key in expected:
        assert key in DEFAULT_MODELS, f"Missing default model key: {key}"
