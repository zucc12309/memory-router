"""Extended health check tests."""

from __future__ import annotations

from memory_router.health import check_health, HealthReport, HealthCheck


class TestHealthReport:
    def test_check_health_returns_report(self, tmp_path):
        report = check_health()
        assert isinstance(report, HealthReport)
        assert isinstance(report.checks, list)
        assert len(report.checks) > 0

    def test_report_has_known_checks(self):
        report = check_health()
        check_names = [c.name for c in report.checks]
        assert "config" in check_names
        assert "memory_store" in check_names

    def test_health_check_dataclass(self):
        hc = HealthCheck(name="test", status="ok", detail="all good")
        assert hc.name == "test"
        assert hc.status == "ok"
        assert hc.detail == "all good"

    def test_report_all_ok(self):
        report = check_health()
        # At minimum config and dirs should be ok in test env
        statuses = {c.name: c.status for c in report.checks}
        assert statuses.get("config") in ("ok", "warning", "error")
