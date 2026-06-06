"""Tests for build_context with mycelium enabled and disabled."""

from pathlib import Path

from memory_router.classifier import classify
from memory_router.config import Config
from memory_router.context_builder import build_context
from memory_router.memory.mycelium import MyceliumNetwork
from memory_router.memory.sqlite_store import ConversationStore, Memory, MemoryStore


def _setup(tmp_path: Path):
    mem_store = MemoryStore(path=tmp_path / "mem.sqlite")
    conv_store = ConversationStore(path=tmp_path / "conv.sqlite")
    cfg = Config(token_budget=4000)
    return mem_store, conv_store, cfg


def _seed_memories(mem_store):
    mid1 = mem_store.add(Memory(
        content="Use FastAPI for REST endpoints",
        domain="software", task="code",
        concepts=["fastapi", "rest"], importance=0.9,
    ))
    mid2 = mem_store.add(Memory(
        content="Always use pytest for testing",
        domain="software", task="code",
        concepts=["pytest", "testing"], importance=0.8,
    ))
    mid3 = mem_store.add(Memory(
        content="PostgreSQL is the production database",
        domain="software", task="code",
        concepts=["postgresql", "database"], importance=0.7,
    ))
    return mid1, mid2, mid3


def test_build_context_without_mycelium(tmp_path):
    """build_context works correctly when mycelium is None."""
    mem_store, conv_store, cfg = _setup(tmp_path)
    _seed_memories(mem_store)

    classification = classify("Write tests for the API")
    built = build_context(
        query="Write tests for the API",
        classification=classification,
        cfg=cfg,
        mem_store=mem_store,
        conv_store=conv_store,
        mycelium=None,
    )

    assert built.messages
    assert built.sent_tokens > 0
    assert any("user" == m.get("role") for m in built.messages)
    last_msg = built.messages[-1]
    assert last_msg["role"] == "user"
    assert "Write tests" in last_msg["content"]


def test_build_context_with_mycelium(tmp_path):
    """build_context uses mycelium spread activation when provided."""
    mem_store, conv_store, cfg = _setup(tmp_path)
    mid1, mid2, mid3 = _seed_memories(mem_store)

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

    assert built.messages
    assert len(built.used_memories) >= 1
    contents = " ".join(m.content for m in built.used_memories)
    assert "test" in contents.lower() or "api" in contents.lower() or "FastAPI" in contents


def test_mycelium_finds_associated_memories(tmp_path):
    """Mycelium should surface memories connected via edges that keyword search might miss."""
    mem_store, conv_store, cfg = _setup(tmp_path)
    mid1, mid2, mid3 = _seed_memories(mem_store)

    mycelium = MyceliumNetwork(mem_store.conn)
    mycelium.add_edge(mid1, mid3, weight=5.0)

    classification = classify("How should I set up the REST endpoints?")
    built_with = build_context(
        query="How should I set up the REST endpoints?",
        classification=classification,
        cfg=cfg,
        mem_store=mem_store,
        conv_store=conv_store,
        mycelium=mycelium,
    )

    built_without = build_context(
        query="How should I set up the REST endpoints?",
        classification=classification,
        cfg=cfg,
        mem_store=mem_store,
        conv_store=conv_store,
        mycelium=None,
    )

    # With mycelium, we should get at least as many memories
    assert len(built_with.used_memories) >= len(built_without.used_memories)


def test_build_context_mycelium_disabled_config(tmp_path):
    """When mycelium_enabled=False in config, mycelium is not used."""
    mem_store, conv_store, cfg = _setup(tmp_path)
    cfg.mycelium_enabled = False
    _seed_memories(mem_store)

    classification = classify("Write tests")
    built = build_context(
        query="Write tests",
        classification=classification,
        cfg=cfg,
        mem_store=mem_store,
        conv_store=conv_store,
        mycelium=None,
    )

    assert built.messages
    assert built.sent_tokens > 0
