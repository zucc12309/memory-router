"""Export engine: SQLite memories → Obsidian Markdown.

ObsidianExporter is the orchestrator. It never reads the vault for retrieval
and never mutates SQLite — it is a one-way projection. All exports are
idempotent and incremental via VaultIndex content hashing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..sqlite_store import Memory, MemoryStore
from .index import VaultIndex
from .models import (
    ConceptLink,
    ExportResult,
    KnowledgeNote,
    category_for,
)
from .renderer import memory_to_markdown, render_knowledge_note
from .utils import sanitize_filename
from .vault import ObsidianVault


class ObsidianExporter:
    """Projects the Memory Palace into an Obsidian vault."""

    def __init__(
        self,
        store: MemoryStore,
        vault: ObsidianVault,
        cfg,
        mycelium=None,
        index: Optional[VaultIndex] = None,
    ):
        self.store = store
        self.vault = vault
        self.cfg = cfg
        self.mycelium = mycelium
        self.index = index or VaultIndex(vault.path)
        self._redact = bool(getattr(cfg, "obsidian_redact_sensitive_data", True))
        self._threshold = float(getattr(cfg, "obsidian_edge_threshold", 0.6))
        self._backlinks = bool(getattr(cfg, "obsidian_generate_backlinks", True))

    # -- public API ---------------------------------------------------------

    def export_all(self) -> ExportResult:
        """Export knowledge notes plus raw memory notes."""
        result = self.export_knowledge_notes()
        result.merge(self.export_raw_memories())
        self.index.save()
        return result

    def export_knowledge_notes(self) -> ExportResult:
        """Consolidate memories by (domain → task) into knowledge notes."""
        result = ExportResult()
        groups = self._group_memories(self.store.list_all(limit=10_000))
        for (domain, _task), mems in groups.items():
            note = self._build_knowledge_note(domain, mems)
            self._write(note.rel_path, render_knowledge_note(note), result)
            result.knowledge_notes_exported += 1
        self.index.save()
        return result

    def export_raw_memories(self) -> ExportResult:
        """Emit one lossless note per memory under 90_Raw_Memories/."""
        result = ExportResult()
        for mem in self.store.list_all(limit=100_000):
            rel_path, content, redactions = self._render_raw(mem)
            self._write(rel_path, content, result)
            result.raw_memories_exported += 1
            result.redactions += redactions
        self.index.save()
        return result

    def export_project(self, project: str) -> ExportResult:
        """Export a single project's knowledge note + its source memories."""
        result = ExportResult()
        needle = project.lower()
        mems = [m for m in self.store.list_all(limit=100_000) if self._matches(m, needle)]
        if not mems:
            result.warnings.append(f"No memories matched project '{project}'.")
            return result
        domain = mems[0].domain
        note = self._build_knowledge_note(domain, mems, title=project)
        self._write(note.rel_path, render_knowledge_note(note), result)
        result.knowledge_notes_exported += 1
        for mem in mems:
            rel_path, content, redactions = self._render_raw(mem)
            self._write(rel_path, content, result)
            result.raw_memories_exported += 1
            result.redactions += redactions
        self.index.save()
        return result

    # -- knowledge note construction ---------------------------------------

    def _group_memories(
        self, mems: List[Memory]
    ) -> Dict[Tuple[str, str], List[Memory]]:
        groups: Dict[Tuple[str, str], List[Memory]] = defaultdict(list)
        for mem in mems:
            groups[(mem.domain, mem.task)].append(mem)
        return groups

    def _build_knowledge_note(
        self, domain: str, mems: List[Memory], title: Optional[str] = None
    ) -> KnowledgeNote:
        title = title or self._title_for(domain, mems)
        ranked = sorted(
            mems, key=lambda m: m.importance * m.confidence, reverse=True
        )
        summary = self._summarize(ranked)
        learnings = [self._oneliner(m) for m in ranked[:8]]
        concepts = self._related_concepts(mems)
        tags = sorted({c for m in mems for c in (m.concepts or [])})
        return KnowledgeNote(
            title=title,
            category=category_for(domain),
            domain=domain,
            summary=summary,
            key_learnings=learnings,
            related_concepts=concepts,
            source_memory_ids=[m.id for m in ranked if m.id is not None],
            tags=tags,
        )

    def _related_concepts(self, mems: List[Memory]) -> List[ConceptLink]:
        """Concept wikilinks, weighted by mycelium edges where available."""
        weights: Dict[str, float] = defaultdict(float)
        # Base weight: concept frequency across the group.
        for mem in mems:
            for concept in mem.concepts or []:
                weights[concept] = max(weights[concept], 0.1)
        # Lift weights using mycelium neighbor strengths.
        if self.mycelium is not None:
            id_to_concepts = {
                m.id: (m.concepts or []) for m in mems if m.id is not None
            }
            for mem in mems:
                if mem.id is None:
                    continue
                for nid, weight, _etype in self.mycelium.get_neighbors(
                    mem.id, min_weight=self._threshold, limit=20
                ):
                    for concept in id_to_concepts.get(nid, []):
                        weights[concept] = max(weights[concept], float(weight))
        links = [
            ConceptLink(concept=c, weight=round(w, 3))
            for c, w in weights.items()
            if w >= self._threshold or w <= 0.1  # keep frequency links + strong edges
        ]
        links.sort(key=lambda x: x.weight, reverse=True)
        return links

    # -- raw note rendering -------------------------------------------------

    def _render_raw(self, mem: Memory) -> Tuple[str, str, int]:
        neighbors = self._neighbors_for(mem)
        content, redactions = memory_to_markdown(
            mem,
            neighbors=neighbors,
            redact=self._redact,
            generate_backlinks=self._backlinks,
        )
        slug = sanitize_filename(f"mem_{mem.id}-{self._slug_hint(mem)}")
        return f"90_Raw_Memories/{slug}.md", content, redactions

    def _neighbors_for(self, mem: Memory) -> List[Tuple[int, float]]:
        if self.mycelium is None or mem.id is None or not self._backlinks:
            return []
        return [
            (nid, weight)
            for nid, weight, _ in self.mycelium.get_neighbors(
                mem.id, min_weight=self._threshold, limit=15
            )
        ]

    # -- write + change detection ------------------------------------------

    def _write(self, rel_path: str, content: str, result: ExportResult) -> None:
        if self.index.is_unchanged(rel_path, content):
            result.notes_skipped += 1
            return
        existed = self.vault.exists(rel_path)
        try:
            self.vault.write_note(rel_path, content, backup=True)
        except ValueError as e:  # path traversal refused
            result.warnings.append(str(e))
            return
        self.index.record(rel_path, content)
        if existed:
            result.notes_updated += 1
        else:
            result.notes_created += 1

    # -- small helpers ------------------------------------------------------

    @staticmethod
    def _matches(mem: Memory, needle: str) -> bool:
        if needle in (mem.domain or "").lower():
            return True
        if needle in (mem.task or "").lower():
            return True
        return any(needle in (c or "").lower() for c in (mem.concepts or []))

    @staticmethod
    def _title_for(domain: str, mems: List[Memory]) -> str:
        concepts = [c for m in mems for c in (m.concepts or [])]
        if concepts:
            top = max(set(concepts), key=concepts.count)
            return top
        return domain.title() if domain else "General"

    @staticmethod
    def _summarize(ranked: List[Memory]) -> str:
        if not ranked:
            return ""
        lead = ranked[0].content.strip().split("\n", 1)[0]
        return f"{lead} (consolidated from {len(ranked)} memories)."

    @staticmethod
    def _oneliner(mem: Memory) -> str:
        text = " ".join((mem.content or "").split())
        return text[:140] + ("…" if len(text) > 140 else "")

    @staticmethod
    def _slug_hint(mem: Memory) -> str:
        if mem.concepts:
            return mem.concepts[0]
        return (mem.content or mem.domain or "memory")[:40]
