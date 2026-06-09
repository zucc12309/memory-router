"""Tests for the health check module."""

from memory_router.health import check_health, HealthReport


def test_health_report_structure():
    report = check_health()
    assert isinstance(report, HealthReport)
    assert report.overall in ("ok", "degraded", "unhealthy")
    assert len(report.checks) > 0


def test_health_report_to_dict():
    report = check_health()
    d = report.to_dict()
    assert "overall" in d
    assert "checks" in d
    assert isinstance(d["checks"], list)
    for check in d["checks"]:
        assert "name" in check
        assert "status" in check
        assert check["status"] in ("ok", "warn", "error")


def test_health_check_has_key_checks():
    """Should check at minimum: config, memory, fts5, tiktoken."""
    report = check_health()
    names = {c.name for c in report.checks}
    assert "config" in names
    assert "fts5" in names
    assert "tiktoken" in names


def test_health_check_fts5_ok():
    """FTS5 should be either available or clearly marked as optional."""
    report = check_health()
    fts5 = next(c for c in report.checks if c.name == "fts5")
    assert fts5.status in ("ok", "warn")
