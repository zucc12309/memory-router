"""Tests for FTS5 query sanitization."""

from pathlib import Path

from memory_router.memory.sqlite_store import MemoryStore, Memory, _sanitize_fts_term


def test_sanitize_fts_term_normal():
    assert _sanitize_fts_term("python") == '"python"'
    assert _sanitize_fts_term("testing") == '"testing"'


def test_sanitize_fts_term_short():
    assert _sanitize_fts_term("ab") == ""
    assert _sanitize_fts_term("") == ""


def test_sanitize_fts_term_non_alpha():
    assert _sanitize_fts_term("test123") == ""
    assert _sanitize_fts_term("hello-world") == ""


def test_sanitize_fts_term_special_chars():
    assert _sanitize_fts_term("NEAR") == '"NEAR"'
    assert _sanitize_fts_term("NOT") == '"NOT"'


def test_search_with_fts_special_chars(tmp_path):
    """Search should not crash with FTS5 special characters in query."""
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    store.add(Memory(content="Python is great for testing", domain="software",
                     task="code", concepts=["python"], importance=0.9))

    # These contain FTS5 operators that could crash unquoted queries
    results = store.search(query_text='NEAR(python, testing)', limit=5)
    assert isinstance(results, list)

    results = store.search(query_text='python NOT testing', limit=5)
    assert isinstance(results, list)

    results = store.search(query_text='python*', limit=5)
    assert isinstance(results, list)


def test_find_similar_with_special_chars(tmp_path):
    """find_similar should not crash with special characters."""
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    store.add(Memory(content="Python testing framework", domain="software",
                     task="code", concepts=["python"], importance=0.9))

    results = store.find_similar('test "injection" content')
    assert isinstance(results, list)
