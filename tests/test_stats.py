"""Stats DB sanity tests.

We patch the STATS_DB path to a tmp_path so we don't pollute the user's
real stats while running tests.
"""

from __future__ import annotations

import importlib

import memory_router.stats as stats_mod


def _isolate(tmp_path, monkeypatch):
    """Repoint stats.sqlite into tmp_path for the duration of the test."""
    monkeypatch.setattr(stats_mod, "STATS_DB", tmp_path / "stats.sqlite")


def test_record_and_summarize(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    s = stats_mod.summarize_stats()
    assert s.calls == 0
    assert s.tokens_saved == 0

    stats_mod.record_usage(
        kind="cli_ask",
        naive_tokens=5000,
        sent_tokens=600,
        output_tokens=900,
        memories_used=3,
        provider="gemini",
        model="gemini-2.5-flash",
        cost_usd=0.0007,
    )
    stats_mod.record_usage(
        kind="mcp_build_context",
        naive_tokens=3200,
        sent_tokens=420,
        memories_used=2,
    )

    s = stats_mod.summarize_stats()
    assert s.calls == 2
    assert s.naive_tokens == 8200
    assert s.sent_tokens == 1020
    assert s.tokens_saved == 7180
    assert s.saved_pct == round(100 * 7180 / 8200)
    assert s.memories_used == 5
    assert "gemini" in s.by_provider
    assert s.by_provider["gemini"]["calls"] == 1
    assert "cli_ask" in s.by_kind and "mcp_build_context" in s.by_kind


def test_record_usage_swallows_errors(tmp_path, monkeypatch):
    """Stats writes must never raise — they're best-effort."""
    _isolate(tmp_path, monkeypatch)
    # Pass a bad cost type — the cast should still succeed but malformed args
    # shouldn't bubble up if anything inside fails.
    stats_mod.record_usage(kind="cli_ask", naive_tokens="not-a-number")  # type: ignore[arg-type]
    s = stats_mod.summarize_stats()
    # Either the row got coerced to 0 or was skipped — both are acceptable.
    assert s.calls in (0, 1)


def test_reset(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    stats_mod.record_usage(kind="cli_ask", naive_tokens=100, sent_tokens=10)
    assert stats_mod.summarize_stats().calls == 1
    n = stats_mod.reset_stats()
    assert n == 1
    assert stats_mod.summarize_stats().calls == 0
