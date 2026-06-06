"""Tests for working memory."""

from memory_router.memory.working_memory import WorkingMemory


def test_put_and_get():
    wm = WorkingMemory(capacity=5)
    wm.put("file", "main.py")
    assert wm.get("file") == "main.py"
    assert wm.size == 1


def test_eviction_on_capacity():
    wm = WorkingMemory(capacity=3)
    wm.put("a", "1", relevance=0.5)
    wm.put("b", "2", relevance=0.9)
    wm.put("c", "3", relevance=0.7)
    # Now at capacity — adding "d" should evict lowest relevance ("a")
    wm.put("d", "4", relevance=0.8)
    assert wm.size == 3
    assert wm.get("a") is None
    assert wm.get("b") == "2"
    assert wm.get("d") == "4"


def test_relevance_decay_on_advance():
    wm = WorkingMemory(capacity=5)
    wm.put("x", "val", relevance=1.0)
    wm.advance_turn()
    wm.advance_turn()
    slots = wm.active_slots(min_relevance=0.0)
    assert len(slots) == 1
    assert slots[0].relevance < 1.0


def test_snapshot_for_context():
    wm = WorkingMemory(capacity=5)
    assert wm.snapshot_for_context() == ""
    wm.put("current_file", "auth.py")
    wm.put("error", "TypeError: missing arg")
    snap = wm.snapshot_for_context()
    assert "current_file" in snap
    assert "auth.py" in snap
    assert "error" in snap


def test_remove():
    wm = WorkingMemory(capacity=5)
    wm.put("a", "1")
    assert wm.remove("a") is True
    assert wm.remove("a") is False
    assert wm.size == 0


def test_clear():
    wm = WorkingMemory(capacity=5)
    wm.put("a", "1")
    wm.put("b", "2")
    wm.clear()
    assert wm.size == 0


def test_update_existing_key():
    wm = WorkingMemory(capacity=5)
    wm.put("file", "old.py")
    wm.put("file", "new.py")
    assert wm.get("file") == "new.py"
    assert wm.size == 1


def test_to_dict():
    wm = WorkingMemory(capacity=5)
    wm.put("k", "v")
    d = wm.to_dict()
    assert d["capacity"] == 5
    assert len(d["slots"]) == 1
    assert d["slots"][0]["key"] == "k"
