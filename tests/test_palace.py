"""Tests for the Memory Palace facade."""

from pathlib import Path

from memory_router.memory.sqlite_store import Memory, MemoryStore
from memory_router.memory.palace import build_palace, PalaceNode


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "memories.sqlite")


def test_build_palace_groups_by_domain_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="use pytest", domain="software", task="code"))
    store.add(Memory(content="prefer dark mode", domain="prefs", task="general"))
    store.add(Memory(content="use typescript", domain="software", task="code"))

    nodes = build_palace(store)
    domains = {n.domain for n in nodes}
    assert "software" in domains
    assert "prefs" in domains

    sw_node = next(n for n in nodes if n.domain == "software")
    assert "code" in sw_node.tasks
    assert len(sw_node.tasks["code"]) == 2


def test_build_palace_empty_store(tmp_path):
    store = _make_store(tmp_path)
    nodes = build_palace(store)
    assert nodes == []


def test_palace_node_type(tmp_path):
    store = _make_store(tmp_path)
    store.add(Memory(content="some fact", domain="test", task="verify"))
    nodes = build_palace(store)
    assert len(nodes) == 1
    assert isinstance(nodes[0], PalaceNode)
