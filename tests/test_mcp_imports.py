"""Test that MCP server module imports and tools register correctly."""

import pytest


def test_mcp_server_module_imports():
    """The mcp_server module should import without requiring the mcp package."""
    from memory_router import mcp_server
    assert hasattr(mcp_server, "_create_server")
    assert hasattr(mcp_server, "main")
    assert hasattr(mcp_server, "_sanitize_session_id")
    assert hasattr(mcp_server, "_sanitize_text")
    assert hasattr(mcp_server, "_check_rate_limit")


def test_mcp_server_create_requires_mcp_package():
    """_create_server should raise RuntimeError if mcp package is missing."""
    try:
        import mcp  # noqa: F401
        pytest.skip("mcp package is installed")
    except ImportError:
        pass

    from memory_router.mcp_server import _create_server
    with pytest.raises(RuntimeError, match="mcp.*not installed"):
        _create_server()


def test_sanitize_session_id():
    from memory_router.mcp_server import _sanitize_session_id

    assert _sanitize_session_id("default") == "default"
    assert _sanitize_session_id("my-session_123") == "my-session_123"
    assert _sanitize_session_id("  ") == "default"
    assert _sanitize_session_id("") == "default"

    with pytest.raises(ValueError):
        _sanitize_session_id("invalid session with spaces!")

    with pytest.raises(ValueError):
        _sanitize_session_id("a" * 200)


def test_sanitize_text():
    from memory_router.mcp_server import _sanitize_text

    assert _sanitize_text("hello") == "hello"
    assert _sanitize_text("  hello  ") == "hello"
    long = "x" * 200_000
    assert len(_sanitize_text(long)) == 100_000
