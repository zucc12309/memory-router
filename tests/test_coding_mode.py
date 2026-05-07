from __future__ import annotations

from memory_router.classifier import Classification
from memory_router.config import Config
from memory_router.context_builder import build_context
from memory_router.memory.auto_capture import capture_turn
from memory_router.memory.sqlite_store import ConversationStore, MemoryStore


def test_build_context_adds_coding_guidance(tmp_path):
    mem_store = MemoryStore(path=tmp_path / "memories.sqlite")
    conv_store = ConversationStore(path=tmp_path / "conversations.sqlite")
    cfg = Config(memory_enabled=False)
    classification = Classification(
        task="code",
        domain="software",
        concepts=["pytest"],
        complexity=0.6,
    )

    built = build_context(
        query="How should we structure tests?",
        classification=classification,
        cfg=cfg,
        mem_store=mem_store,
        conv_store=conv_store,
        use_memory=False,
    )

    assert built.messages[0]["role"] == "system"
    assert "Coding mode" in built.messages[0]["content"]


def test_capture_turn_stores_stable_coding_preferences(tmp_path):
    store = MemoryStore(path=tmp_path / "memories.sqlite")
    cfg = Config(memory_enabled=True, auto_capture_memories=True)
    classification = Classification(
        task="code",
        domain="software",
        concepts=["pytest"],
        complexity=0.7,
    )

    memory_id = capture_turn(
        query="For this repo, always use pytest for tests.",
        answer="Got it. Use pytest and local-only tests.",
        classification=classification,
        cfg=cfg,
        store=store,
    )

    assert memory_id is not None
    memories = store.list_all()
    assert len(memories) == 1
    assert "pytest" in memories[0].content.lower()
