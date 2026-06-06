"""Tests for memory import/export functionality."""

import json
from pathlib import Path

from memory_router.memory.sqlite_store import Memory, MemoryStore
from memory_router.memory.importer import (
    export_memories,
    export_to_file,
    import_from_file,
)


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "memories.sqlite")


def test_export_and_reimport_native(tmp_path):
    """Round-trip: export then import should reproduce memories."""
    store = _make_store(tmp_path)
    store.add(Memory(content="fact one", domain="code", importance=0.8))
    store.add(Memory(content="fact two", domain="ops", importance=0.6))

    out_path = tmp_path / "export.json"
    count = export_to_file(store, out_path)
    assert count == 2

    # Import into a fresh store
    store2 = MemoryStore(path=tmp_path / "memories2.sqlite")
    imported, skipped = import_from_file(store2, out_path)
    assert imported == 2
    assert skipped == 0
    assert store2.count() == 2


def test_import_deduplicates(tmp_path):
    """Importing the same file twice should skip duplicates."""
    store = _make_store(tmp_path)
    store.add(Memory(content="existing fact", importance=0.5))

    data = [{"content": "existing fact"}, {"content": "new fact here"}]
    path = tmp_path / "generic.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    imported, skipped = import_from_file(store, path)
    assert imported == 1  # Only new fact
    assert skipped == 1  # existing fact skipped


def test_import_chatgpt_format(tmp_path):
    """Import from ChatGPT conversations.json format."""
    chatgpt_data = [
        {
            "title": "Test Conversation",
            "mapping": {
                "node-1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["This is a user message for testing import"]}
                    }
                },
                "node-2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["This is an assistant reply"]}
                    }
                },
            },
        }
    ]

    path = tmp_path / "conversations.json"
    path.write_text(json.dumps(chatgpt_data), encoding="utf-8")

    store = _make_store(tmp_path)
    imported, skipped = import_from_file(store, path)
    assert imported == 1  # Only user messages
    assert skipped == 0

    mems = store.list_all()
    assert len(mems) == 1
    assert "user message" in mems[0].content
    assert mems[0].memory_type == "episodic"


def test_import_claude_format(tmp_path):
    """Import from Claude conversation export format."""
    claude_data = [
        {
            "uuid": "abc-123",
            "chat_messages": [
                {"sender": "human", "text": "Tell me about memory systems and how they work in detail"},
                {"sender": "assistant", "text": "Memory systems involve..."},
                {"sender": "human", "text": "Can you explain more about episodic memory specifically"},
            ],
        }
    ]

    path = tmp_path / "claude_export.json"
    path.write_text(json.dumps(claude_data), encoding="utf-8")

    store = _make_store(tmp_path)
    imported, skipped = import_from_file(store, path)
    assert imported == 2  # Two human messages
    assert skipped == 0


def test_import_generic_list(tmp_path):
    """Import from a generic list of strings/objects."""
    data = [
        "Remember that the user prefers dark mode in all applications",
        {"content": "The API endpoint is /v2/users for user management", "domain": "api", "importance": 0.9},
        "tiny",  # Too short (<5 chars), should be skipped
    ]

    path = tmp_path / "generic.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    store = _make_store(tmp_path)
    imported, skipped = import_from_file(store, path)
    assert imported == 2  # The two valid items
    assert skipped == 0

    mems = store.list_all()
    domains = {m.domain for m in mems}
    assert "api" in domains


def test_import_skips_short_chatgpt_messages(tmp_path):
    """ChatGPT messages under 20 chars should be skipped."""
    chatgpt_data = {
        "conversations": [
            {
                "title": "Short chat",
                "mapping": {
                    "n1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Hi"]}
                        }
                    }
                },
            }
        ]
    }

    path = tmp_path / "short.json"
    path.write_text(json.dumps(chatgpt_data), encoding="utf-8")

    store = _make_store(tmp_path)
    imported, _ = import_from_file(store, path)
    assert imported == 0


def test_export_memories_fields(tmp_path):
    """Exported memories should include all important fields."""
    store = _make_store(tmp_path)
    store.add(Memory(
        content="test content",
        domain="test",
        task="validation",
        concepts=["testing", "export"],
        importance=0.7,
        memory_type="procedural",
        source="unit_test",
    ))

    exported = export_memories(store)
    assert len(exported) == 1
    m = exported[0]
    assert m["content"] == "test content"
    assert m["domain"] == "test"
    assert m["concepts"] == ["testing", "export"]
    assert m["memory_type"] == "procedural"
    assert m["importance"] == 0.7
