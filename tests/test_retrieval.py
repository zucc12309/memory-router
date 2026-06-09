"""Tests for the retrieval module."""


from memory_router.classifier import classify
from memory_router.memory.retrieval import retrieve_relevant_memories
from memory_router.memory.sqlite_store import Memory, MemoryStore
from memory_router.memory.mycelium import MyceliumNetwork


def test_retrieve_without_mycelium(tmp_path):
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    store.add(Memory(content="Use pytest for testing", domain="software",
                     task="code", concepts=["pytest"], importance=0.9))

    cls = classify("Write tests for the module")
    results = retrieve_relevant_memories(
        store=store, classification=cls, query="Write tests for the module",
        limit=5, mycelium=None, touch=False,
    )
    assert len(results) >= 1
    assert any("pytest" in m.content for m in results)


def test_retrieve_with_mycelium(tmp_path):
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    mid1 = store.add(Memory(content="Use FastAPI", domain="software",
                            task="code", concepts=["fastapi"], importance=0.9))
    mid2 = store.add(Memory(content="Use PostgreSQL", domain="software",
                            task="code", concepts=["postgresql"], importance=0.7))

    mycelium = MyceliumNetwork(store.conn)
    mycelium.add_edge(mid1, mid2, weight=3.0)

    cls = classify("Set up the API endpoints")
    results = retrieve_relevant_memories(
        store=store, classification=cls, query="Set up the API endpoints",
        limit=5, mycelium=mycelium, touch=False,
    )
    assert len(results) >= 1


def test_retrieve_touches_memories(tmp_path):
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    mid = store.add(Memory(content="Important fact about Python", domain="software",
                           task="code", concepts=["python"], importance=0.9))
    mem_before = store.get(mid)
    assert mem_before.usage_count == 0

    cls = classify("Tell me about Python")
    retrieve_relevant_memories(
        store=store, classification=cls, query="Tell me about Python",
        limit=5, mycelium=None, touch=True,
    )

    mem_after = store.get(mid)
    assert mem_after.usage_count >= 1


def test_retrieve_empty_store(tmp_path):
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    cls = classify("anything")
    results = retrieve_relevant_memories(
        store=store, classification=cls, query="anything",
        limit=5, mycelium=None,
    )
    assert results == []
