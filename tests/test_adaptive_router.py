"""Tests for the adaptive router with outcome learning."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from memory_router.adaptive_router import (
    AdaptiveRouter,
    RouteOutcome,
)
from memory_router.classifier import Classification
from memory_router.config import Config


def _make_router(tmp_path: Path, mode: str = "hybrid") -> AdaptiveRouter:
    """Create an AdaptiveRouter using a temp database."""
    cfg = Config()
    cfg.mode = mode
    router = AdaptiveRouter(cfg)
    # Point to temp DB so tests don't pollute real data
    db_path = tmp_path / "route_history.sqlite"
    import memory_router.adaptive_router as ar_mod
    ar_mod.ROUTE_HISTORY_DB = db_path
    router._conn = None  # Force re-init with new path
    return router


def _classification(task="code", domain="general", complexity=0.5):
    return Classification(task=task, domain=domain, concepts=[], complexity=complexity)


def test_record_outcome(tmp_path):
    router = _make_router(tmp_path)
    router.record_outcome(RouteOutcome(
        provider="openai",
        model="gpt-4o-mini",
        task="code",
        domain="general",
        complexity=0.3,
        input_tokens=100,
        output_tokens=50,
        latency_ms=500,
        cost_usd=0.001,
        quality_signal=0.8,
    ))

    # Should appear in history
    report = router.get_performance_report(min_samples=1)
    assert len(report) == 1
    assert report[0].provider == "openai"
    assert report[0].model == "gpt-4o-mini"
    assert report[0].avg_quality == 0.8


def test_fallback_to_rule_based_with_insufficient_data(tmp_path):
    """With fewer than MIN_SAMPLES, should fall back to rule-based routing."""
    router = _make_router(tmp_path)

    # Record only 2 outcomes (below MIN_SAMPLES=5)
    for _ in range(2):
        router.record_outcome(RouteOutcome(
            provider="openai", model="gpt-4o-mini",
            task="code", domain="general", complexity=0.3,
        ))

    # Should fall back to rule-based — not crash
    with patch.object(router._rule_router, "route") as mock_route:
        mock_route.return_value = MagicMock()
        router.route(_classification())
        mock_route.assert_called_once()


def test_adaptive_routing_with_sufficient_data(tmp_path):
    """With enough samples, should use adaptive routing."""
    router = _make_router(tmp_path)

    # Create a mock provider that's available
    mock_provider = MagicMock()
    mock_provider.is_available.return_value = True
    router._rule_router.providers["openai"] = mock_provider

    # Record 6 outcomes (above MIN_SAMPLES=5)
    for i in range(6):
        router.record_outcome(RouteOutcome(
            provider="openai", model="gpt-4o-mini",
            task="code", domain="general", complexity=0.3,
            quality_signal=0.85, cost_usd=0.001, latency_ms=400,
        ))

    result = router.route(_classification(task="code"))
    assert result is not None
    assert "adaptive" in result.reason


def test_performance_report(tmp_path):
    router = _make_router(tmp_path)

    for i in range(3):
        router.record_outcome(RouteOutcome(
            provider="openai", model="gpt-4o",
            task="chat", domain="general", complexity=0.2,
            quality_signal=0.9, cost_usd=0.005, latency_ms=800,
        ))

    report = router.get_performance_report(min_samples=1)
    assert len(report) == 1
    perf = report[0]
    assert perf.provider == "openai"
    assert perf.avg_quality == 0.9
    assert perf.sample_count == 3


def test_reset_history(tmp_path):
    router = _make_router(tmp_path)

    for _ in range(3):
        router.record_outcome(RouteOutcome(
            provider="openai", model="gpt-4o",
            task="chat", domain="general", complexity=0.2,
        ))

    deleted = router.reset_history()
    assert deleted == 3
    assert router.get_performance_report(min_samples=1) == []


def test_error_outcomes_excluded_from_adaptive(tmp_path):
    """Outcomes with errors should not count toward adaptive routing."""
    router = _make_router(tmp_path)

    mock_provider = MagicMock()
    mock_provider.is_available.return_value = True
    router._rule_router.providers["openai"] = mock_provider

    # Record 6 error outcomes
    for _ in range(6):
        router.record_outcome(RouteOutcome(
            provider="openai", model="gpt-4o-mini",
            task="code", domain="general", complexity=0.3,
            error="timeout",
        ))

    # Should fall back to rule-based since all outcomes have errors
    with patch.object(router._rule_router, "route") as mock_route:
        mock_route.return_value = MagicMock()
        router.route(_classification())
        mock_route.assert_called_once()


def test_override_bypasses_adaptive(tmp_path):
    """Provider/model overrides should skip adaptive logic."""
    router = _make_router(tmp_path)

    with patch.object(router._rule_router, "route") as mock_route:
        mock_route.return_value = MagicMock()
        router.route(_classification(), override_provider="anthropic")
        mock_route.assert_called_once()
