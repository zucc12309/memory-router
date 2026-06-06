"""Tests for memory consolidation (near-duplicate merging)."""

from pathlib import Path

from memory_router.memory.sqlite_store import Memory, MemoryStore
from memory_router.memory.consolidation import consolidate_memories


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "memories.sqlite")


def test_no_duplicates_returns_zero(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="the sky is blue today", importance=0.5))
    store.add(Memory(content="python uses indentation for blocks", importance=0.7))

    result = consolidate_memories(store, similarity_threshold=0.6)
    assert result.clusters_found == 0
    assert result.memories_merged == 0


def test_merges_near_duplicates(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="user prefers dark mode in all applications", importance=0.8))
    store.add(Memory(content="user prefers dark mode in applications always", importance=0.5))
    store.add(Memory(content="something completely different about python coding", importance=0.6))

    result = consolidate_memories(store, similarity_threshold=0.5, dry_run=False)
    assert result.clusters_found == 1
    assert result.memories_merged == 1
    assert store.count() == 2  # anchor + unrelated


def test_dry_run_doesnt_change(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="user prefers dark mode in all applications", importance=0.8))
    store.add(Memory(content="user prefers dark mode in applications always", importance=0.5))

    result = consolidate_memories(store, similarity_threshold=0.5, dry_run=True)
    assert result.clusters_found == 1
    assert result.memories_merged == 1
    assert store.count() == 2  # Nothing deleted in dry run


def test_anchor_keeps_highest_importance(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="the api endpoint is at slash v2 slash users", importance=0.3))
    store.add(Memory(content="the api endpoint is at slash v2 slash users resource", importance=0.9))

    consolidate_memories(store, similarity_threshold=0.5, dry_run=False)
    remaining = store.list_all()
    assert len(remaining) == 1
    assert remaining[0].importance >= 0.9  # Kept the high-importance one


def test_concepts_merged_from_cluster(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(
        content="user prefers dark mode in all applications",
        importance=0.8, concepts=["dark-mode", "preferences"],
    ))
    store.add(Memory(
        content="user prefers dark mode in applications always",
        importance=0.5, concepts=["ui", "theme"],
    ))

    consolidate_memories(store, similarity_threshold=0.5, dry_run=False)
    remaining = store.list_all()
    assert len(remaining) == 1
    # All concepts should be merged
    concepts = set(remaining[0].concepts)
    assert "dark-mode" in concepts
    assert "ui" in concepts
