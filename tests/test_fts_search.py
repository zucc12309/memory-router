"""Tests for FTS5 full-text search in MemoryStore."""

from pathlib import Path

from memory_router.memory.sqlite_store import Memory, MemoryStore


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "memories.sqlite")


def test_fts_search_finds_content(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="Always use pytest for testing", domain="software", task="code",
                     concepts=["pytest", "testing"], importance=0.9))
    store.add(Memory(content="Prefer TypeScript over JavaScript", domain="software", task="code",
                     concepts=["typescript", "javascript"], importance=0.8))
    store.add(Memory(content="Bond convexity measures curvature", domain="finance", task="explain",
                     concepts=["bond", "convexity"], importance=0.7))

    # Search with query text — should find pytest memory
    results = store.search(query_text="pytest", limit=5)
    assert len(results) >= 1
    assert any("pytest" in m.content.lower() for m in results)


def test_search_with_domain_filter(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="Use FastAPI", domain="software", task="code"))
    store.add(Memory(content="Bond duration", domain="finance", task="explain"))

    results = store.search(domain="software", limit=5)
    assert all(m.domain == "software" for m in results)


def test_search_ranks_stack_memory_for_project_stack_question(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="Prefer pytest for testing", domain="software", task="code",
                     concepts=["pytest", "tests"], importance=0.95))
    store.add(Memory(content="This project uses TypeScript and pnpm", domain="software", task="code",
                     concepts=["typescript", "pnpm"], importance=0.9))
    store.add(Memory(content="User likes concise answers", domain="prefs", task="general",
                     concepts=["concise", "answers"], importance=0.99))

    results = store.search(query_text="Which stack does the project use?", limit=3)
    assert len(results) >= 2
    assert results[0].content.startswith("This project uses TypeScript")
    assert all(m.domain == "software" for m in results[:2])


def test_search_ranks_testing_memory_for_test_framework_question(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="Prefer pytest for testing", domain="software", task="code",
                     concepts=["pytest", "tests"], importance=0.95))
    store.add(Memory(content="This project uses TypeScript and pnpm", domain="software", task="code",
                     concepts=["typescript", "pnpm"], importance=0.9))
    store.add(Memory(content="User likes concise answers", domain="prefs", task="general",
                     concepts=["concise", "answers"], importance=0.99))

    results = store.search(query_text="What test framework should I use in this repo?", limit=3)
    assert len(results) >= 2
    assert results[0].content.startswith("Prefer pytest")
    assert all(m.domain == "software" for m in results[:2])


def test_search_by_type(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="Fact A", memory_type="semantic"))
    store.add(Memory(content="Event B", memory_type="episodic"))
    store.add(Memory(content="How to C", memory_type="procedural"))

    semantic = store.search_by_type("semantic")
    assert len(semantic) == 1
    assert semantic[0].content == "Fact A"


def test_new_columns_present(tmp_path):
    store = _make_store(tmp_path)
    mid = store.add(Memory(
        content="test",
        confidence=0.8,
        memory_type="procedural",
        source="agent",
    ))
    mem = store.get(mid)
    assert mem is not None
    assert mem.confidence == 0.8
    assert mem.memory_type == "procedural"
    assert mem.source == "agent"


def test_count(tmp_path):
    store = _make_store(tmp_path)
    assert store.count() == 0
    store.add(Memory(content="a"))
    store.add(Memory(content="b"))
    assert store.count() == 2


def test_update_importance(tmp_path):
    store = _make_store(tmp_path)
    mid = store.add(Memory(content="test", importance=0.5))
    store.update_importance(mid, 0.9)
    mem = store.get(mid)
    assert mem.importance == 0.9


def test_cached_token_count(tmp_path):
    from memory_router.memory.sqlite_store import ConversationStore, Message

    cs = ConversationStore(path=tmp_path / "conv.sqlite")
    assert cs.count_tokens_for_session() == 0
    cs.add(Message(session_id="default", role="user", content="Hello world"))
    count = cs.count_tokens_for_session()
    assert count > 0


def test_session_count(tmp_path):
    from memory_router.memory.sqlite_store import ConversationStore, Message

    cs = ConversationStore(path=tmp_path / "conv.sqlite")
    assert cs.session_count() == 0
    cs.add(Message(session_id="default", role="user", content="hi"))
    assert cs.session_count() == 1
