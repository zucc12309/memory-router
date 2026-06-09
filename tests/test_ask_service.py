"""Tests for AskService orchestration and quality signal heuristic."""

from __future__ import annotations


from memory_router.ask_service import AskService, AskResult, _estimate_quality
from memory_router.config import Config
from memory_router.memory.sqlite_store import MemoryStore, ConversationStore


class TestEstimateQuality:
    def test_error_gives_zero(self):
        assert _estimate_quality("", "connection refused", 1000) == 0.0

    def test_empty_answer_low_quality(self):
        assert _estimate_quality("", None, 1000) == 0.2

    def test_short_answer(self):
        q = _estimate_quality("yes", None, 500)
        assert q == 0.2  # < 10 chars

    def test_medium_answer(self):
        q = _estimate_quality("x" * 100, None, 500)
        assert 0.5 <= q <= 0.8

    def test_long_answer(self):
        q = _estimate_quality("x" * 2000, None, 500)
        assert q >= 0.8

    def test_high_latency_penalty(self):
        fast = _estimate_quality("x" * 500, None, 1000)
        slow = _estimate_quality("x" * 500, None, 35000)
        assert slow < fast

    def test_quality_bounded(self):
        q = _estimate_quality("x" * 100000, None, 100)
        assert q <= 1.0


class TestAskServiceInit:
    def test_creates_with_config(self):
        cfg = Config()
        svc = AskService(cfg)
        assert svc.cfg is cfg

    def test_lazy_store_creation(self, tmp_path):
        cfg = Config()
        mem = MemoryStore(path=tmp_path / "m.sqlite")
        conv = ConversationStore(path=tmp_path / "c.sqlite")
        svc = AskService(cfg, mem_store=mem, conv_store=conv)
        assert svc.mem_store is mem
        assert svc.conv_store is conv


class TestAskResult:
    def test_has_request_id(self):
        r = AskResult(
            answer="hello",
            classification=None,
            route_decision_reason="test",
            provider="test",
            model="test",
            input_tokens=10,
            output_tokens=5,
            full_history_tokens=100,
            sent_tokens=10,
            latency_ms=100,
            cost_usd=0.001,
            token_savings_pct=90.0,
            memories_used=0,
        )
        assert len(r.request_id) == 12
        assert r.fallback_used is False
        assert r.error is None
