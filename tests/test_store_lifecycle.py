"""Tests for store lifecycle (close, context manager) and validation."""

import pytest
from pathlib import Path

from memory_router.memory.sqlite_store import (
    ConversationStore,
    Memory,
    MemoryStore,
    Message,
    _VALID_MEMORY_TYPES,
    _VALID_SOURCES,
)


def test_memory_store_close(tmp_path):
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    store.add(Memory(content="test", importance=0.5))
    store.close()


def test_memory_store_context_manager(tmp_path):
    with MemoryStore(path=tmp_path / "mem.sqlite") as store:
        mid = store.add(Memory(content="test", importance=0.5))
        assert mid > 0


def test_conversation_store_close(tmp_path):
    store = ConversationStore(path=tmp_path / "conv.sqlite")
    store.add(Message(session_id="s1", role="user", content="hello"))
    store.close()


def test_conversation_store_context_manager(tmp_path):
    with ConversationStore(path=tmp_path / "conv.sqlite") as store:
        mid = store.add(Message(session_id="s1", role="user", content="hello"))
        assert mid > 0


def test_invalid_memory_type_rejected(tmp_path):
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    with pytest.raises(ValueError, match="Invalid memory_type"):
        store.add(Memory(content="test", memory_type="sematic"))


def test_invalid_source_rejected(tmp_path):
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    with pytest.raises(ValueError, match="Invalid source"):
        store.add(Memory(content="test", source="unknown"))


def test_valid_memory_types(tmp_path):
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    for mtype in _VALID_MEMORY_TYPES:
        mid = store.add(Memory(content=f"test {mtype}", memory_type=mtype))
        assert mid > 0


def test_valid_sources(tmp_path):
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    for source in _VALID_SOURCES:
        mid = store.add(Memory(content=f"test {source}", source=source))
        assert mid > 0
