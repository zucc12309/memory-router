"""Obsidian export layer for Memory Router.

A one-way, human-readable projection of the SQLite Memory Palace into an
Obsidian vault. SQLite remains the source of truth; retrieval never touches
the vault. Disabled by default — opt in with ``obsidian_enabled``.

Public surface::

    from memory_router.memory.obsidian import ObsidianVault, ObsidianExporter

    vault = ObsidianVault(cfg.obsidian_vault_path).init()
    exporter = ObsidianExporter(store, vault, cfg, mycelium=myc)
    result = exporter.export_all()
"""

from __future__ import annotations

from .exporter import ObsidianExporter
from .index import VaultIndex
from .models import ConceptLink, ExportResult, KnowledgeNote, category_for
from .renderer import (
    markdown_to_memory,
    memory_to_markdown,
    render_knowledge_note,
    wikilink,
)
from .vault import ObsidianVault

__all__ = [
    "ObsidianVault",
    "ObsidianExporter",
    "VaultIndex",
    "ExportResult",
    "KnowledgeNote",
    "ConceptLink",
    "category_for",
    "memory_to_markdown",
    "markdown_to_memory",
    "render_knowledge_note",
    "wikilink",
]
