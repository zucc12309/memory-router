"""Tests for the upgraded context builder with priority scoring."""

from pathlib import Path

from memory_router.classifier import classify
from memory_router.config import Config
from memory_router.context_builder import build_context
from memory_router.memory.sqlite_store import (
    ConversationStore,
    Memory,
    MemoryStore,
    Message,
)
from memory_router.memory.working_memory import WorkingMemory


def _setup(tmp_path: Path):
    mem_store = MemoryStore(path=tmp_path / "mem.sqlite")
    conv_store = ConversationStore(path=tmp_path / "conv.sqlite")
    cfg = Config(token_budget=4000)
    return mem_store, conv_store, cfg


def test_basic_build(tmp_path):
    mem_store, conv_store, cfg = _setup(tmp_path)
    mem_store.add(Memory(content="User prefers pytest", domain="software",
                         task="code", concepts=["pytest"], importance=0.9))

    classification = classify("Write tests for the parser")
    built = build_context(
        query="Write tests for the parser",
        classification=classification,
        cfg=cfg,
        mem_store=mem_store,
        conv_store=conv_store,
    )

    assert built.messages
    assert built.sent_tokens > 0
    assert any("pytest" in m.get("content", "") for m in built.messages)


def test_working_memory_included(tmp_path):
    mem_store, conv_store, cfg = _setup(tmp_path)
    wm = WorkingMemory(capacity=10)
    wm.put("current_file", "auth.py")

    classification = classify("Fix the authentication bug")
    built = build_context(
        query="Fix the authentication bug",
        classification=classification,
        cfg=cfg,
        mem_store=mem_store,
        conv_store=conv_store,
        working_memory=wm,
    )

    # Working memory snapshot should appear in the messages
    has_wm = any("auth.py" in m.get("content", "") for m in built.messages)
    assert has_wm


def test_priority_preserves_query(tmp_path):
    mem_store, conv_store, cfg = _setup(tmp_path)
    cfg.token_budget = 100  # Very tight budget

    classification = classify("Important question")
    built = build_context(
        query="Important question about something",
        classification=classification,
        cfg=cfg,
        mem_store=mem_store,
        conv_store=conv_store,
    )

    # The query should always be preserved even with tight budget
    assert any(
        "Important question" in m.get("content", "") for m in built.messages
    )


def test_mycelium_integration(tmp_path):
    """Test that mycelium spread activation is used when provided."""
    from memory_router.memory.mycelium import MyceliumNetwork

    mem_store, conv_store, cfg = _setup(tmp_path)
    mid1 = mem_store.add(Memory(content="Use FastAPI", domain="software",
                                task="code", concepts=["fastapi"], importance=0.9))
    mid2 = mem_store.add(Memory(content="Use pytest", domain="software",
                                task="code", concepts=["pytest"], importance=0.8))
    mid3 = mem_store.add(Memory(content="PostgreSQL for DB", domain="software",
                                task="code", concepts=["postgresql"], importance=0.7))

    # Create mycelium edges between memories
    mycelium = MyceliumNetwork(mem_store.conn)
    mycelium.add_edge(mid1, mid2, weight=3.0)
    mycelium.add_edge(mid2, mid3, weight=2.0)

    classification = classify("Write tests for the API")
    built = build_context(
        query="Write tests for the API",
        classification=classification,
        cfg=cfg,
        mem_store=mem_store,
        conv_store=conv_store,
        mycelium=mycelium,
    )

    # Should find memories through mycelium spread
    assert len(built.used_memories) >= 1


def test_untrusted_context_is_not_system_role(tmp_path):
    mem_store, conv_store, cfg = _setup(tmp_path)
    mem_store.add(Memory(content="Prefer pytest", domain="software", task="code", concepts=["pytest"], importance=0.9))
    for i in range(8):
        conv_store.add(Message(session_id="default", role="user", content=f"Earlier question {i}"))
        conv_store.add(Message(session_id="default", role="assistant", content=f"Earlier answer {i}"))
    wm = WorkingMemory(capacity=10)
    wm.put("current_file", "auth.py")

    classification = classify("Write tests for auth.py")
    built = build_context(
        query="Write tests for auth.py",
        classification=classification,
        cfg=cfg,
        mem_store=mem_store,
        conv_store=conv_store,
        working_memory=wm,
    )

    system_contents = [m["content"] for m in built.messages if m["role"] == "system"]
    user_contents = [m["content"] for m in built.messages if m["role"] == "user"]

    assert any("Coding mode:" in text for text in system_contents)
    assert any("Memory notes" in text for text in user_contents)
    assert any("Working memory" in text for text in user_contents)
    assert any("Conversation summary" in text for text in user_contents)
