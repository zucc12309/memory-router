"""Configuration management for Memory Router.

Handles the local config directory, YAML config file, and runtime defaults.
Everything is stored under ~/.memory-router/ — nothing leaves the machine.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# ---------- paths ----------

ROOT_DIR = Path(os.path.expanduser("~/.memory-router"))
CONFIG_PATH = ROOT_DIR / "config.yaml"
CONVERSATIONS_DB = ROOT_DIR / "conversations.sqlite"
MEMORIES_DB = ROOT_DIR / "memories.sqlite"
VECTOR_DIR = ROOT_DIR / "vector_index"
LOG_DIR = ROOT_DIR / "logs"


# ---------- defaults ----------

DEFAULT_MODELS = {
    "local_simple": "llama3.2:3b",
    "local_default": "llama3.1:8b",
    "openai_small": "gpt-4o-mini",
    "openai_large": "gpt-4o",
    "anthropic_small": "claude-haiku-4-5-20251001",
    "anthropic_mid": "claude-sonnet-4-6",
    "anthropic_large": "claude-opus-4-7",
    "gemini_small": "gemini-2.5-flash",
    "gemini_mid": "gemini-2.5-pro",
    "gemini_large": "gemini-2.5-pro",
}


@dataclass
class Config:
    """User-facing config, persisted to ~/.memory-router/config.yaml."""

    mode: str = "local"  # local | api | hybrid | ruflo
    default_provider: str = "ollama"  # openai | anthropic | ollama | ruflo | gemini
    ollama_host: str = "http://localhost:11434"
    memory_enabled: bool = True
    auto_capture_memories: bool = True
    max_recent_messages: int = 6
    max_relevant_memories: int = 4
    token_budget: int = 4000
    models: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    # Pin a specific provider+model and skip auto-routing entirely.
    # Either is empty string for "auto-pick". Override per-call with --provider/--model.
    force_provider: str = ""   # e.g. "gemini" | "openai" | "anthropic" | "ollama"
    force_model: str = ""      # e.g. "gemini-2.5-flash" | "gpt-4o-mini"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        # Merge unknown keys safely; only known fields are taken.
        defaults = cls()
        for key in defaults.__dataclass_fields__:
            if key in data:
                setattr(defaults, key, data[key])
        # Backfill any model tier keys that weren't in the saved file (e.g.
        # the user's config predates Gemini support). User overrides win.
        merged_models = dict(DEFAULT_MODELS)
        merged_models.update(defaults.models or {})
        defaults.models = merged_models
        return defaults


# ---------- io ----------

def ensure_dirs() -> None:
    """Create the local storage tree if missing. Safe to call repeatedly."""
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in (ROOT_DIR, VECTOR_DIR, LOG_DIR):
        try:
            os.chmod(path, stat.S_IRWXU)
        except Exception:
            pass


def _lock_file(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


def load_config() -> Config:
    """Load config from disk, returning defaults if no file exists yet."""
    if not CONFIG_PATH.exists():
        return Config()
    _lock_file(CONFIG_PATH)
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.from_dict(data)


def save_config(cfg: Config) -> None:
    """Persist config to ~/.memory-router/config.yaml."""
    ensure_dirs()
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    _lock_file(CONFIG_PATH)


def is_initialized() -> bool:
    return CONFIG_PATH.exists()


def set_value(key: str, value: Any) -> Config:
    """Update a single config field. Used by `memory-router config set`."""
    cfg = load_config()
    if key not in cfg.__dataclass_fields__:
        raise KeyError(f"Unknown config key: {key}")
    # Cast booleans/ints from strings since CLI passes raw strings.
    current = getattr(cfg, key)
    if isinstance(current, bool):
        value = str(value).lower() in ("1", "true", "yes", "on")
    elif isinstance(current, int):
        value = int(value)
    setattr(cfg, key, value)
    save_config(cfg)
    return cfg
