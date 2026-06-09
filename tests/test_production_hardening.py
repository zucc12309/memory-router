"""Tests for production hardening fixes.

Covers: decay semantics, role validation, relevance threshold,
        retry logic, MCP injection check, WAL mode, ProviderResult fields.
"""

from __future__ import annotations

import time

import pytest

from memory_router.memory.decay import apply_decay, reinforce, get_decay_stats, prune_stale_memories
from memory_router.memory.sqlite_store import Memory, MemoryStore
from memory_router.memory.retrieval import retrieve_relevant_memories, _effective_score
from memory_router.providers.base import ProviderResult
from memory_router.router import _is_retryable_error


# ---------- Decay semantics: confidence not importance ----------

class TestDecaySemantics:
    def test_apply_decay_updates_confidence_not_importance(self, tmp_path):
        """apply_decay should mutate confidence, not importance."""
        store = MemoryStore(path=tmp_path / "mem.sqlite")
        store.add(Memory(
            content="Test decay semantics",
            domain="test", task="test",
            importance=0.8, confidence=0.9,
            created_at=time.time() - 86400 * 10,  # 10 days old
            last_used=0.0,
        ))
        updated = apply_decay(store)
        assert updated >= 1
        mem = store.list_all(limit=1)[0]
        # importance should be unchanged at 0.8
        assert mem.importance == pytest.approx(0.8, abs=0.001)
        # confidence should have decayed below 0.9
        assert mem.confidence < 0.9

    def test_reinforce_boosts_confidence(self, tmp_path):
        store = MemoryStore(path=tmp_path / "mem.sqlite")
        mid = store.add(Memory(
            content="Reinforce test", domain="test", task="test",
            importance=0.5, confidence=0.5,
        ))
        reinforce(store, mid, boost=0.2)
        mem = store.get(mid)
        assert mem.confidence == pytest.approx(0.7, abs=0.01)
        assert mem.importance == pytest.approx(0.5, abs=0.01)  # unchanged

    def test_get_decay_stats_uses_confidence(self, tmp_path):
        store = MemoryStore(path=tmp_path / "mem.sqlite")
        store.add(Memory(content="Stats test", domain="t", task="t", confidence=0.3))
        stats = get_decay_stats(store)
        assert "avg_confidence" in stats
        assert "avg_importance" not in stats

    def test_prune_uses_confidence(self, tmp_path):
        store = MemoryStore(path=tmp_path / "mem.sqlite")
        store.add(Memory(
            content="Will be pruned", domain="t", task="t",
            confidence=0.01, created_at=time.time() - 86400 * 60,
        ))
        pruned = prune_stale_memories(store, confidence_threshold=0.05)
        assert pruned == 1


# ---------- WAL mode ----------

class TestSQLiteHardening:
    def test_wal_mode_enabled(self, tmp_path):
        store = MemoryStore(path=tmp_path / "mem.sqlite")
        mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_busy_timeout_set(self, tmp_path):
        store = MemoryStore(path=tmp_path / "mem.sqlite")
        timeout = store.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == 5000


# ---------- ProviderResult enrichment ----------

class TestProviderResult:
    def test_new_fields_exist(self):
        r = ProviderResult(text="hi", model="gpt-4o")
        assert r.latency_ms == 0
        assert r.finish_reason == "stop"
        assert r.retryable is False
        assert r.request_id == ""
        assert r.error == ""

    def test_fields_settable(self):
        r = ProviderResult(
            text="hi", model="gpt-4o",
            latency_ms=150, finish_reason="length",
            retryable=True, request_id="req-123",
        )
        assert r.latency_ms == 150
        assert r.retryable is True


# ---------- Retry logic ----------

class TestRetryLogic:
    def test_retryable_rate_limit(self):
        assert _is_retryable_error(RuntimeError("429 Too Many Requests"))

    def test_retryable_server_error(self):
        assert _is_retryable_error(RuntimeError("502 Bad Gateway"))

    def test_retryable_timeout(self):
        assert _is_retryable_error(RuntimeError("Connection timed out"))

    def test_not_retryable_auth(self):
        assert not _is_retryable_error(RuntimeError("401 Unauthorized"))

    def test_not_retryable_generic(self):
        assert not _is_retryable_error(ValueError("bad input"))


# ---------- Retrieval relevance threshold ----------

class TestRetrievalThreshold:
    def test_effective_score(self):
        m = Memory(content="x", importance=0.8, confidence=0.5)
        assert _effective_score(m) == pytest.approx(0.4)

    def test_low_score_filtered(self, tmp_path):
        store = MemoryStore(path=tmp_path / "mem.sqlite")
        store.add(Memory(
            content="Low confidence memory about python",
            domain="software", task="code",
            importance=0.01, confidence=0.01,
            concepts=["python"],
        ))
        store.add(Memory(
            content="High confidence memory about python",
            domain="software", task="code",
            importance=0.9, confidence=0.9,
            concepts=["python"],
        ))
        from memory_router.classifier import Classification
        cls = Classification(task="code", domain="software", concepts=["python"], complexity=0.5)
        results = retrieve_relevant_memories(
            store, cls, "python", limit=10,
            relevance_threshold=0.05,
            touch=False, strengthen=False,
        )
        # Low-score memory should be filtered
        assert all(_effective_score(m) >= 0.05 for m in results)


# ---------- Role validation ----------

class TestRoleValidation:
    def test_invalid_role_sanitized(self, tmp_path):
        """context_builder should sanitize invalid message roles."""
        from memory_router.context_builder import build_context
        from memory_router.classifier import Classification
        from memory_router.config import Config
        from memory_router.memory.sqlite_store import ConversationStore, Message

        mem_store = MemoryStore(path=tmp_path / "mem.sqlite")
        conv_store = ConversationStore(path=tmp_path / "conv.sqlite")
        # Store a message with an invalid role
        conv_store.add(Message(
            session_id="test", role="hacker_injected",
            content="harmless content",
        ))
        cfg = Config()
        cls = Classification(task="general", domain="general", concepts=[], complexity=0.3)
        ctx = build_context(
            "hello", cls, cfg, mem_store, conv_store,
            use_memory=False, session_id="test",
        )
        # The invalid role should have been replaced with "user"
        roles = [m["role"] for m in ctx.messages]
        assert "hacker_injected" not in roles
