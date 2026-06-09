"""Health check module for Memory Router.

Provides a single `check_health()` function that returns a structured
report of system status — useful for `memory-router doctor`, MCP health
tools, and programmatic integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class HealthCheck:
    name: str
    status: str  # "ok" | "warn" | "error"
    detail: str = ""


@dataclass
class HealthReport:
    overall: str = "ok"  # "ok" | "degraded" | "unhealthy"
    checks: List[HealthCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
        }


def check_health() -> HealthReport:
    """Run all health checks and return a structured report."""
    checks: List[HealthCheck] = []

    # 1. Config file
    checks.append(_check_config())

    # 2. SQLite stores
    checks.append(_check_memory_store())
    checks.append(_check_conversation_store())

    # 3. FTS5
    checks.append(_check_fts5())

    # 4. Provider availability
    checks.extend(_check_providers())

    # 5. Optional dependencies
    checks.append(_check_tiktoken())
    checks.append(_check_encryption())

    # 6. Memory health
    checks.append(_check_memory_health())

    # Determine overall status
    statuses = [c.status for c in checks]
    if "error" in statuses:
        overall = "unhealthy"
    elif "warn" in statuses:
        overall = "degraded"
    else:
        overall = "ok"

    return HealthReport(overall=overall, checks=checks)


def _check_config() -> HealthCheck:
    try:
        from .config import is_initialized, load_config

        if not is_initialized():
            return HealthCheck("config", "warn", "Not initialized. Run: memory-router init")
        cfg = load_config()
        return HealthCheck("config", "ok", f"mode={cfg.mode}")
    except Exception as e:
        return HealthCheck("config", "error", str(e))


def _check_memory_store() -> HealthCheck:
    try:
        from .memory.sqlite_store import MemoryStore

        store = MemoryStore()
        count = store.count()
        return HealthCheck("memory_store", "ok", f"{count} memories")
    except Exception as e:
        return HealthCheck("memory_store", "error", str(e))


def _check_conversation_store() -> HealthCheck:
    try:
        from .memory.sqlite_store import ConversationStore

        ConversationStore()  # verify we can instantiate
        return HealthCheck("conversation_store", "ok", "connected")
    except Exception as e:
        return HealthCheck("conversation_store", "error", str(e))


def _check_fts5() -> HealthCheck:
    try:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE _fts_test USING fts5(content)")
        conn.execute("DROP TABLE _fts_test")
        conn.close()
        return HealthCheck("fts5", "ok", "supported")
    except Exception:
        return HealthCheck("fts5", "warn", "FTS5 not available — search will be slower")


def _check_providers() -> List[HealthCheck]:
    checks = []
    try:
        from .providers.ollama_provider import OllamaProvider

        ollama = OllamaProvider()
        if ollama.is_available():
            checks.append(HealthCheck("ollama", "ok", "connected"))
        else:
            checks.append(HealthCheck("ollama", "warn", "not running"))
    except Exception:
        checks.append(HealthCheck("ollama", "warn", "not available"))

    for name in ("openai", "anthropic", "gemini"):
        try:
            from .security.keychain import get_secret

            key = get_secret(name)
            if key:
                checks.append(HealthCheck(name, "ok", "API key configured"))
            else:
                checks.append(HealthCheck(name, "warn", "no API key"))
        except Exception:
            checks.append(HealthCheck(name, "warn", "check failed"))

    return checks


def _check_tiktoken() -> HealthCheck:
    try:
        import tiktoken  # noqa: F401

        return HealthCheck("tiktoken", "ok", "installed — accurate token counting")
    except ImportError:
        return HealthCheck("tiktoken", "warn", "not installed — using heuristic estimation")


def _check_encryption() -> HealthCheck:
    try:
        from .security.encryption import is_encryption_available

        if is_encryption_available():
            return HealthCheck("encryption", "ok", "AES-256-GCM available")
        return HealthCheck("encryption", "warn", "cryptography not installed")
    except Exception:
        return HealthCheck("encryption", "warn", "check failed")


def _check_memory_health() -> HealthCheck:
    try:
        from .memory.sqlite_store import MemoryStore
        from .memory.decay import get_decay_stats

        store = MemoryStore()
        stats = get_decay_stats(store)
        total = stats["total_memories"]
        stale = stats["stale_count"]
        if total == 0:
            return HealthCheck("memory_health", "ok", "no memories yet")
        stale_pct = (stale / total) * 100
        if stale_pct > 50:
            return HealthCheck(
                "memory_health", "warn",
                f"{stale_pct:.0f}% stale — run: memory-router memory decay --prune"
            )
        return HealthCheck("memory_health", "ok", f"{total} memories, {stale} stale")
    except Exception as e:
        return HealthCheck("memory_health", "warn", str(e))
