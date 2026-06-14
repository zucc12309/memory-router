"""Data models and constants for the Obsidian export layer.

These are projection models — they describe Markdown notes generated *from*
SQLite, never an alternative source of truth. The canonical model stays
``memory_router.memory.sqlite_store.Memory``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Vault layout
# ---------------------------------------------------------------------------

# Folders created by ObsidianVault.init(). Order matters only for readability.
VAULT_FOLDERS: List[str] = [
    "00_Inbox",
    "01_Projects",
    "02_Research",
    "03_Decisions",
    "04_People",
    "05_Conversations",
    "06_Daily",
    "90_Raw_Memories",
    "99_Archive",
]

INDEX_NOTE = "Memory Router Index.md"
README_NOTE = "README.md"
GITIGNORE = ".gitignore"
MANIFEST = ".memory-router-index.json"

# Domain keyword → knowledge-note folder. Matched by substring, first hit wins.
# Covers every domain the rule-based classifier can emit (finance/software/ml/
# legal/medical/science) plus common free-form tags.
_CATEGORY_RULES: List[Tuple[Tuple[str, ...], str]] = [
    (("project", "app", "product", "software", "code", "engineering", "dev"), "01_Projects"),
    (
        ("research", "science", "ml", "ai", "paper", "study", "math",
         "finance", "legal", "medical", "physics", "biology", "chemistry"),
        "02_Research",
    ),
    (("decision", "architecture", "adr", "tradeoff", "policy", "agentic"), "03_Decisions"),
    (("people", "person", "contact", "team", "user"), "04_People"),
    (("conversation", "chat", "thread", "session"), "05_Conversations"),
]

# Fallback for domain="general" (the classifier's default): route by *task* so
# generic memories don't all pile into the Inbox.
_TASK_FALLBACK: List[Tuple[Tuple[str, ...], str]] = [
    (("code", "test", "debug"), "01_Projects"),
    (("explain", "summarize", "reasoning", "research"), "02_Research"),
    (("agentic", "plan", "design", "decision"), "03_Decisions"),
    (("chat", "conversation"), "05_Conversations"),
]


def category_for(domain: str, task: str = "") -> str:
    """Map a memory's (domain, task) to a knowledge-note folder.

    Domain is the primary signal. When it is empty or "general" (the
    classifier's default), fall back to the task so generic memories still
    get sorted instead of all landing in 00_Inbox.
    """
    d = (domain or "").lower()
    if d and d != "general":
        for keywords, folder in _CATEGORY_RULES:
            if any(k in d for k in keywords):
                return folder
    t = (task or "").lower()
    if t:
        for keywords, folder in _TASK_FALLBACK:
            if any(k in t for k in keywords):
                return folder
    return "00_Inbox"


# ---------------------------------------------------------------------------
# Projection models
# ---------------------------------------------------------------------------


@dataclass
class ConceptLink:
    """A wikilink target with an optional mycelium edge weight."""

    concept: str
    weight: float = 0.0


@dataclass
class KnowledgeNote:
    """A consolidated, human-readable note derived from many memories.

    Knowledge notes are lossy projections (summaries + links). They are NOT
    round-tripped back into SQLite — only raw memory notes are.
    """

    title: str
    category: str  # vault folder, e.g. "01_Projects"
    domain: str
    summary: str = ""
    key_learnings: List[str] = field(default_factory=list)
    related_concepts: List[ConceptLink] = field(default_factory=list)
    source_memory_ids: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def rel_path(self) -> str:
        from .utils import sanitize_filename

        return f"{self.category}/{sanitize_filename(self.title)}.md"


@dataclass
class ExportResult:
    """Summary of one export run. Returned by ObsidianExporter methods."""

    notes_created: int = 0
    notes_updated: int = 0
    notes_skipped: int = 0
    knowledge_notes_exported: int = 0
    raw_memories_exported: int = 0
    redactions: int = 0
    warnings: List[str] = field(default_factory=list)

    def merge(self, other: "ExportResult") -> "ExportResult":
        """Accumulate another result into this one (in place) and return self."""
        self.notes_created += other.notes_created
        self.notes_updated += other.notes_updated
        self.notes_skipped += other.notes_skipped
        self.knowledge_notes_exported += other.knowledge_notes_exported
        self.raw_memories_exported += other.raw_memories_exported
        self.redactions += other.redactions
        self.warnings.extend(other.warnings)
        return self

    def to_dict(self) -> Dict[str, object]:
        return {
            "notes_created": self.notes_created,
            "notes_updated": self.notes_updated,
            "notes_skipped": self.notes_skipped,
            "knowledge_notes_exported": self.knowledge_notes_exported,
            "raw_memories_exported": self.raw_memories_exported,
            "redactions": self.redactions,
            "warnings": self.warnings,
        }
