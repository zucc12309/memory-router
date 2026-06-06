from __future__ import annotations

from memory_router.cli import _memory_preview, _memory_summary, _message_border_style
from memory_router.memory.sqlite_store import Memory


def test_memory_preview_formats_metadata_and_snippet():
    mem = Memory(
        task="code",
        domain="software",
        concepts=["pytest", "cli", "mcp"],
        content="Prefer pytest for backend and MCP tests with concise fixtures.",
        memory_type="semantic",
    )

    preview = _memory_preview(mem, content_limit=48)

    assert "software/code" in preview
    assert "(semantic)" in preview
    assert "[pytest, cli, ...]" in preview
    assert preview.endswith("...")


def test_memory_summary_handles_empty_and_multiple_memories():
    first = Memory(domain="software", task="code", content="Prefer pytest for backend and MCP tests.")
    second = Memory(domain="research", task="general", content="Keep API key usage offline when possible.")

    assert _memory_summary([]) == "none"

    summary = _memory_summary([first, second])
    assert "2 memories used" in summary
    assert "software/code" in summary
    assert "(+1 more)" in summary


def test_message_border_style_is_role_specific():
    assert _message_border_style("system") == "blue"
    assert _message_border_style("user") == "green"
    assert _message_border_style("assistant") == "magenta"
    assert _message_border_style("tool") == "cyan"
