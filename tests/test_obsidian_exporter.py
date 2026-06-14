"""Tests for ObsidianExporter: idempotency, knowledge notes, mycelium, projects."""

from __future__ import annotations


from memory_router.config import Config
from memory_router.memory.mycelium import MyceliumNetwork
from memory_router.memory.obsidian import ObsidianExporter, ObsidianVault
from memory_router.memory.obsidian.index import VaultIndex
from memory_router.memory.obsidian.models import ExportResult
from memory_router.memory.sqlite_store import Memory, MemoryStore


def _setup(tmp_path, redact=True, threshold=0.6):
    store = MemoryStore(path=tmp_path / "mem.sqlite")
    ids = [
        store.add(Memory(
            content="RideCompare uses CatBoost for fare prediction",
            domain="software", task="explain",
            concepts=["RideCompare", "CatBoost", "Flutter"], importance=0.9)),
        store.add(Memory(
            content="CatBoost is a gradient boosting library",
            domain="research", task="explain",
            concepts=["CatBoost", "ML"], importance=0.7)),
        store.add(Memory(
            content="Flutter renders the mobile UI",
            domain="software", task="explain",
            concepts=["Flutter", "RideCompare"], importance=0.6)),
    ]
    myc = MyceliumNetwork(store.conn)
    myc.strengthen_co_retrieved(ids, boost=0.8)  # weight 0.8 >= threshold 0.6
    cfg = Config()
    cfg.obsidian_redact_sensitive_data = redact
    cfg.obsidian_edge_threshold = threshold
    vault = ObsidianVault(tmp_path / "vault").init()
    exporter = ObsidianExporter(store, vault, cfg, mycelium=myc)
    return store, myc, cfg, vault, exporter, ids


class TestExportAll:
    def test_creates_knowledge_and_raw(self, tmp_path):
        _, _, _, _, exporter, _ = _setup(tmp_path)
        result = exporter.export_all()
        assert result.knowledge_notes_exported >= 1
        assert result.raw_memories_exported == 3
        assert result.notes_created > 0

    def test_returns_export_result(self, tmp_path):
        _, _, _, _, exporter, _ = _setup(tmp_path)
        assert isinstance(exporter.export_all(), ExportResult)


class TestIdempotency:
    def test_second_run_skips_all(self, tmp_path):
        _, _, _, _, exporter, _ = _setup(tmp_path)
        exporter.export_all()
        r2 = exporter.export_all()
        assert r2.notes_created == 0
        assert r2.notes_updated == 0
        assert r2.notes_skipped > 0

    def test_change_triggers_update(self, tmp_path):
        store, _, _, _, exporter, ids = _setup(tmp_path)
        exporter.export_all()
        # Mutate a memory, re-export → at least one updated note.
        store.conn.execute(
            "UPDATE memories SET content = ? WHERE id = ?",
            ("RideCompare now uses XGBoost instead", ids[0]),
        )
        store.conn.commit()
        r2 = exporter.export_all()
        assert r2.notes_updated >= 1

    def test_index_persists(self, tmp_path):
        _, _, _, vault, exporter, _ = _setup(tmp_path)
        exporter.export_all()
        idx = VaultIndex(vault.path)
        assert idx.note_count > 0
        assert idx.last_export > 0


class TestKnowledgeNotes:
    def test_grouped_into_folders(self, tmp_path):
        _, _, _, vault, exporter, _ = _setup(tmp_path)
        exporter.export_knowledge_notes()
        md_files = list((vault.path / "01_Projects").glob("*.md"))
        md_files += list((vault.path / "02_Research").glob("*.md"))
        assert md_files

    def test_only_knowledge_no_raw(self, tmp_path):
        _, _, _, vault, exporter, _ = _setup(tmp_path)
        result = exporter.export_knowledge_notes()
        assert result.raw_memories_exported == 0
        assert not list((vault.path / "90_Raw_Memories").glob("*.md"))


class TestMyceliumWikilinks:
    def test_edges_become_wikilinks(self, tmp_path):
        _, _, _, vault, exporter, ids = _setup(tmp_path)
        exporter.export_raw_memories()
        matches = list((vault.path / "90_Raw_Memories").glob(f"mem_{ids[0]}-*.md"))
        assert matches, "raw note for first memory should exist"
        note = matches[0].read_text()
        assert "## Linked Memories" in note
        assert "[[mem_" in note
        assert "weight" in note  # edge weight comment present

    def test_threshold_filters_weak_edges(self, tmp_path):
        # Threshold above the edge weight (0.8) → no links emitted.
        _, _, _, vault, exporter, ids = _setup(tmp_path, threshold=5.0)
        exporter.export_raw_memories()
        raw = list((vault.path / "90_Raw_Memories").glob("*.md"))[0].read_text()
        assert "## Linked Memories" not in raw


class TestRedaction:
    def test_secret_redacted_in_export(self, tmp_path):
        store, myc, cfg, vault, exporter, _ = _setup(tmp_path, redact=True)
        store.add(Memory(
            content="token=sk-abcdefghijklmnopqrstuvwxyz0123456789",
            domain="software", task="code", concepts=["secret"]))
        result = exporter.export_raw_memories()
        assert result.redactions >= 1
        for f in (vault.path / "90_Raw_Memories").glob("*.md"):
            assert "sk-abcdefghijklmnopqrstuvwxyz0123456789" not in f.read_text()


class TestExportProject:
    def test_export_specific_project(self, tmp_path):
        _, _, _, vault, exporter, _ = _setup(tmp_path)
        result = exporter.export_project("RideCompare")
        assert result.knowledge_notes_exported == 1
        assert result.raw_memories_exported >= 1

    def test_unknown_project_warns(self, tmp_path):
        _, _, _, _, exporter, _ = _setup(tmp_path)
        result = exporter.export_project("NonexistentThing")
        assert result.warnings
        assert result.knowledge_notes_exported == 0


class TestNoMycelium:
    def test_export_without_mycelium(self, tmp_path):
        store = MemoryStore(path=tmp_path / "m.sqlite")
        store.add(Memory(content="solo memory", domain="general", task="chat",
                         concepts=["x"]))
        cfg = Config()
        vault = ObsidianVault(tmp_path / "v").init()
        exporter = ObsidianExporter(store, vault, cfg, mycelium=None)
        result = exporter.export_all()
        assert result.raw_memories_exported == 1


class TestExportResultMerge:
    def test_merge_accumulates(self):
        a = ExportResult(notes_created=1, knowledge_notes_exported=1, warnings=["w1"])
        b = ExportResult(notes_created=2, raw_memories_exported=3, warnings=["w2"])
        a.merge(b)
        assert a.notes_created == 3
        assert a.raw_memories_exported == 3
        assert a.knowledge_notes_exported == 1
        assert a.warnings == ["w1", "w2"]
