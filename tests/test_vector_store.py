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


def test_add_replace():
    store = _make_store()
    store.add(1, [0.1, 0.2, 0.3, 0.4])
    store.add(1, [0.5, 0.6, 0.7, 0.8])
    assert store.count() == 1


def test_query_zero_vector():
    store = _make_store()
    store.add(1, [1.0, 0.0, 0.0, 0.0])
    results = store.query([0.0, 0.0, 0.0, 0.0])
    assert results == []


def test_query_top_k_limit():
    store = _make_store()
    for i in range(10):
        store.add(i, [float(i), 1.0, 0.0, 0.0])
    results = store.query([5.0, 1.0, 0.0, 0.0], top_k=3)
    assert len(results) <= 3


def test_blob_roundtrip():
    store = _make_store()
    original = [0.123, 0.456, 0.789, 1.011]
    blob = store._to_blob(original)
    restored = store._from_blob(blob)
    assert len(restored) == 4
    for a, b in zip(original, restored):
        assert abs(a - b) < 1e-5


def test_python_fallback():
    store = _make_store()
    store._numpy = None  # Force Python path
    store.add(1, [1.0, 0.0, 0.0, 0.0])
    store.add(2, [0.0, 1.0, 0.0, 0.0])
    rows = store.conn.execute(
        "SELECT memory_id, embedding FROM memory_embeddings"
    ).fetchall()
    hits = store._query_python([1.0, 0.0, 0.0, 0.0], rows, top_k=2)
    assert len(hits) >= 1
    assert hits[0].memory_id == 1


def test_disabled_add_remove_noop():
    store = VectorStore(conn=None, dim=4)
    store.add(1, [0.1, 0.2, 0.3, 0.4])  # no-op
    assert store.count() == 0
    store.remove(1)  # no-op, shouldn't raise
