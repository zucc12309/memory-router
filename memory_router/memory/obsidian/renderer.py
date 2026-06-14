"""Markdown rendering and parsing for the Obsidian layer.

Two note kinds:
  * Raw memory notes — lossless round-trip with ``Memory`` (frontmatter holds
    every scalar field; body holds ``content`` verbatim).
  * Knowledge notes — lossy, human-readable projections (summaries + links).
    These are generated, never parsed back.

Frontmatter is YAML (the ``yaml`` package is already a hard dependency via
``config.py``), so no new third-party requirement is introduced.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import yaml

from ..sqlite_store import Memory
from .models import ConceptLink, KnowledgeNote
from .utils import from_iso, redact as _redact, slugify, to_iso

# Sentinel that separates verbatim memory content from the generated link
# block, so round-trip parsing can recover the exact original content.
_LINK_SENTINEL = "<!-- mr:links -->"
_FRONTMATTER_FENCE = "---"


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


def serialize_frontmatter(data: dict) -> str:
    """Render a YAML frontmatter block (with fences), keys in insertion order."""
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip()
    return f"{_FRONTMATTER_FENCE}\n{body}\n{_FRONTMATTER_FENCE}"


def parse_frontmatter(text: str) -> Tuple[dict, str]:
    """Split a note into (frontmatter_dict, body). Tolerant of no frontmatter."""
    if not text.startswith(_FRONTMATTER_FENCE):
        return {}, text
    lines = text.split("\n")
    # Find the closing fence.
    for i in range(1, len(lines)):
        if lines[i].strip() == _FRONTMATTER_FENCE:
            raw = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :]).lstrip("\n")
            data = yaml.safe_load(raw) or {}
            if not isinstance(data, dict):
                data = {}
            return data, body
    return {}, text


# ---------------------------------------------------------------------------
# Wikilinks
# ---------------------------------------------------------------------------


def wikilink(target: str, weight: Optional[float] = None) -> str:
    """Render a ``[[target]]`` link, annotating edge weight as a comment."""
    link = f"[[{target}]]"
    if weight is not None and weight > 0:
        link += f" <!-- weight {weight:.2f} -->"
    return link


def _mem_link_target(memory_id: int) -> str:
    return f"mem_{memory_id}"


# ---------------------------------------------------------------------------
# Raw memory notes (lossless round-trip)
# ---------------------------------------------------------------------------


def memory_to_markdown(
    mem: Memory,
    neighbors: Optional[List[Tuple[int, float]]] = None,
    *,
    redact: bool = False,
    generate_backlinks: bool = True,
) -> Tuple[str, int]:
    """Render a single ``Memory`` as a raw memory note.

    Returns (markdown, n_redactions). ``neighbors`` is a list of
    (neighbor_memory_id, weight) from the mycelium graph.
    """
    content = mem.content or ""
    redactions = 0
    if redact:
        content, redactions = _redact(content)

    concepts = list(mem.concepts or [])
    fm = {
        "id": mem.id,
        "memory_type": mem.memory_type,
        "source": mem.source,
        "task": mem.task,
        "domain": mem.domain,
        "created_at": to_iso(mem.created_at),
        "last_used": to_iso(mem.last_used),
        "importance": round(float(mem.importance), 4),
        "confidence": round(float(mem.confidence), 4),
        "usage_count": int(mem.usage_count),
        "concepts": concepts,
    }

    title = _memory_title(mem, content)
    parts = [serialize_frontmatter(fm), "", f"# {title}", "", content.rstrip()]

    if generate_backlinks and neighbors:
        parts.append("")
        parts.append(_LINK_SENTINEL)
        parts.append("## Linked Memories")
        parts.append("")
        for nid, weight in neighbors:
            parts.append(f"- {wikilink(_mem_link_target(nid), weight)}")

    return "\n".join(parts) + "\n", redactions


def markdown_to_memory(text: str) -> Memory:
    """Parse a raw memory note back into a ``Memory`` (lossless for content).

    Redacted notes will not recover the original secret — that is intentional;
    SQLite remains the source of truth.
    """
    fm, body = parse_frontmatter(text)

    # Strip the generated link block and the leading display heading.
    content_block = body.split(_LINK_SENTINEL, 1)[0].rstrip()
    content = _strip_leading_h1(content_block)

    concepts = fm.get("concepts") or []
    if isinstance(concepts, str):
        concepts = [concepts]

    return Memory(
        id=fm.get("id"),
        task=str(fm.get("task", "general")),
        domain=str(fm.get("domain", "general")),
        concepts=list(concepts),
        content=content,
        importance=float(fm.get("importance", 0.5)),
        confidence=float(fm.get("confidence", 1.0)),
        memory_type=str(fm.get("memory_type", "semantic")),
        source=str(fm.get("source", "import")),
        created_at=from_iso(fm.get("created_at")),
        last_used=from_iso(fm.get("last_used")),
        usage_count=int(fm.get("usage_count", 0)),
    )


def _memory_title(mem: Memory, content: str) -> str:
    """Derive a stable, readable H1 for a raw memory note."""
    if mem.concepts:
        base = " · ".join(mem.concepts[:3])
    else:
        first_line = (content or "").strip().split("\n", 1)[0]
        base = first_line[:60] or f"{mem.domain}/{mem.task}"
    return base


def _strip_leading_h1(body: str) -> str:
    """Remove a single generated ``# Title`` line + following blank line."""
    lines = body.split("\n")
    if lines and lines[0].startswith("# "):
        rest = lines[1:]
        if rest and rest[0].strip() == "":
            rest = rest[1:]
        return "\n".join(rest).rstrip()
    return body.rstrip()


# ---------------------------------------------------------------------------
# Knowledge notes (generated projection)
# ---------------------------------------------------------------------------


def render_knowledge_note(note: KnowledgeNote) -> str:
    """Render a consolidated, human-readable knowledge note."""
    fm = {
        "type": "knowledge_note",
        "source": "memory-router",
        "domain": note.domain,
        "concepts": [c.concept for c in note.related_concepts] or note.tags,
        "source_memories": [f"mem_{i}" for i in note.source_memory_ids],
    }
    parts = [serialize_frontmatter(fm), "", f"# {note.title}", ""]

    parts.append("## Summary")
    parts.append("")
    parts.append(note.summary or "_No summary available._")
    parts.append("")

    if note.key_learnings:
        parts.append("## Key Learnings")
        parts.append("")
        for item in note.key_learnings:
            parts.append(f"- {item}")
        parts.append("")

    if note.related_concepts:
        parts.append("## Related Concepts")
        parts.append("")
        for link in note.related_concepts:
            parts.append(f"- {wikilink(_concept_target(link), link.weight)}")
        parts.append("")

    if note.source_memory_ids:
        parts.append("## Source Memories")
        parts.append("")
        for mid in note.source_memory_ids:
            parts.append(f"- {wikilink(_mem_link_target(mid))}")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _concept_target(link: ConceptLink) -> str:
    """Wikilink target for a concept — title-cased, human-readable."""
    return link.concept if link.concept else slugify(link.concept)
