"""Memory Router — local-first LLM router with structured Memory Palace.

Quick start::

    from memory_router import MemoryStore, Memory, classify, Config

    store = MemoryStore()
    store.add(Memory(content="User prefers dark mode", domain="prefs"))
    results = store.search(query_text="theme preferences")
"""

__version__ = "0.2.0"

from .classifier import Classification, classify
from .config import Config, load_config, save_config
from .context_builder import BuiltContext, build_context
from .memory.sqlite_store import (
    ConversationStore,
    Memory,
    MemoryStore,
    Message,
)
from .router import RouteDecision, Router
from .health import check_health, HealthReport

__all__ = [
    "__version__",
    # Core types
    "Classification",
    "Config",
    "Memory",
    "MemoryStore",
    "Message",
    "ConversationStore",
    "RouteDecision",
    "BuiltContext",
    "HealthReport",
    # Functions
    "classify",
    "build_context",
    "load_config",
    "save_config",
    "check_health",
    "Router",
]
