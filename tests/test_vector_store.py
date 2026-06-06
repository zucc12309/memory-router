"""Tests for the vector store."""

import sqlite3

from memory_router.memory.vector_store import VectorStore


def _make_store():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return VectorStore(conn=conn, dim=4)


def test_add_and_query():
    store = _make_store()
    assert store.enabled

    store.add(1, [1.0, 0.0, 0.0, 0.0])
    store.add(2, [0.0, 1.0, 0.0, 0.0])
    store.add(3, [0.9, 0.1, 0.0, 0.0])

    results = store.query([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) >= 1
    # Memory 1 should be most similar to [1,0,0,0]
    assert results[0].memory_id == 1
    assert results[0].score > 0.9


def test_remove():
    store = _make_store()
    store.add(1, [1.0, 0.0, 0.0, 0.0])
    assert store.count() == 1
    store.remove(1)
    assert store.count() == 0


def test_empty_query():
    store = _make_store()
    results = store.query([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert results == []


def test_noop_without_connection():
    store = VectorStore(conn=None, dim=4)
    assert not store.enabled
    results = store.query([1.0], top_k=5)
    assert results == []
