"""Tests for memory decay and reinforcement.

Decay operates on the *confidence* field (temporal reliability),
not importance (user-assigned base weight).
"""

import time
from pathlib import Path

from memory_router.memory.sqlite_store import Memory, MemoryStore
from memory_router.memory.decay import (
    apply_decay,
    reinforce,
    prune_stale_memories,
    get_decay_stats,
)


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "memories.sqlite")


def test_reinforce_boosts_confidence(tmp_path):
    store = _make_store(tmp_path)
    mid = store.add(Memory(content="test fact", importance=0.5, confidence=0.5))
    reinforce(store, mid, boost=0.2)
    mem = store.get(mid)
    assert mem is not None
    assert mem.confidence >= 0.7  # confidence boosted
    assert mem.importance == 0.5  # importance unchanged
    assert mem.usage_count == 1


def test_apply_decay_reduces_old_memories(tmp_path):
    store = _make_store(tmp_path)
    # Create a memory that appears old (created 90 days ago, never used)
    old_time = time.time() - (90 * 86400)
    store.add(Memory(content="old fact", importance=0.8, confidence=0.8, created_at=old_time))

    # Apply decay as if checking now
    count = apply_decay(store)
    assert count >= 1

    mems = store.list_all()
    assert len(mems) == 1
    assert mems[0].confidence < 0.8  # Should have decayed
    assert mems[0].importance == 0.8  # importance unchanged


def test_prune_stale_memories(tmp_path):
    store = _make_store(tmp_path)
    old_time = time.time() - (60 * 86400)
    store.add(Memory(content="stale", confidence=0.02, created_at=old_time))
    store.add(Memory(content="fresh", confidence=0.9))

    pruned = prune_stale_memories(store, confidence_threshold=0.05, min_age_days=30)
    assert pruned == 1
    assert store.count() == 1
    assert store.list_all()[0].content == "fresh"


def test_prune_respects_min_age(tmp_path):
    store = _make_store(tmp_path)
    # New memory with low confidence — should NOT be pruned
    store.add(Memory(content="new but low", confidence=0.02))
    pruned = prune_stale_memories(store, confidence_threshold=0.05, min_age_days=30)
    assert pruned == 0
    assert store.count() == 1


def test_get_decay_stats(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="a", confidence=0.9))
    store.add(Memory(content="b", confidence=0.05))
    store.add(Memory(content="c", confidence=0.5))

    stats = get_decay_stats(store)
    assert stats["total_memories"] == 3
    assert stats["stale_count"] == 1  # confidence < 0.1
    assert stats["strong_count"] == 1  # confidence >= 0.8
