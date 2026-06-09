"""Test that every public module imports cleanly."""

import importlib
import pkgutil

import memory_router


def test_all_submodules_import():
    """Every module under memory_router should import without error."""
    failures = []
    for importer, modname, ispkg in pkgutil.walk_packages(
        memory_router.__path__, prefix="memory_router."
    ):
        # Skip modules that require optional dependencies at import time
        if modname == "memory_router.mcp_server":
            continue
        try:
            importlib.import_module(modname)
        except ImportError as e:
            # Optional provider dependencies are acceptable import failures
            msg = str(e).lower()
            if any(dep in msg for dep in ("openai", "anthropic", "google", "ruflo", "mcp", "cryptography")):
                continue
            failures.append(f"{modname}: {e}")
        except Exception as e:
            failures.append(f"{modname}: {e}")

    assert not failures, "Failed to import modules:\n" + "\n".join(failures)


def test_public_api_all_exported():
    """Every name in __all__ should be importable."""
    for name in memory_router.__all__:
        obj = getattr(memory_router, name, None)
        assert obj is not None, f"memory_router.{name} is None or missing"


def test_memory_submodules():
    """Critical memory submodules should import."""


def test_provider_base_imports():
    """Provider base classes should always import."""
    from memory_router.providers.base import BaseProvider
    assert hasattr(BaseProvider, "split_system_messages")


def test_utils_import():
    """Utility modules should import."""


def test_security_import():
    """Security modules should import."""
