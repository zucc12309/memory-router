"""Tests for semantic deduplication (find_similar)."""

from pathlib import Path

from memory_router.memory.sqlite_store import Memory, MemoryStore


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "memories.sqlite")


def test_find_similar_returns_matches(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="the user prefers dark mode in all applications", importance=0.8))
    store.add(Memory(content="python uses whitespace for indentation blocks", importance=0.6))

    similar = store.find_similar("user prefers dark mode in every application", threshold=0.5)
    assert len(similar) >= 1
    assert any("dark mode" in m.content for m in similar)


def test_find_similar_no_matches(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="the user prefers dark mode in all applications", importance=0.8))

    similar = store.find_similar("quantum physics entanglement theory", threshold=0.7)
    assert len(similar) == 0


def test_find_similar_high_threshold(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="python uses indentation for code blocks", importance=0.5))
    store.add(Memory(content="python uses indentation for code block structure", importance=0.6))

    # Very high threshold should match near-exact only
    similar = store.find_similar("python uses indentation for code blocks", threshold=0.9)
    # The exact match should be found
    assert len(similar) >= 1


def test_find_similar_empty_content(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="some memory content here", importance=0.5))

    similar = store.find_similar("", threshold=0.5)
    assert len(similar) == 0
