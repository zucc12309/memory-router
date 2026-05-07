from __future__ import annotations

from memory_router.cli import _rewrite_argv


def test_rewrite_argv_prepends_ask_for_free_form_query_with_flags():
    assert _rewrite_argv(["--no-memory", "Explain bond convexity"]) == [
        "ask",
        "--no-memory",
        "Explain bond convexity",
    ]


def test_rewrite_argv_leaves_known_commands_alone():
    assert _rewrite_argv(["--no-memory", "ask", "Explain bond convexity"]) == [
        "--no-memory",
        "ask",
        "Explain bond convexity",
    ]
