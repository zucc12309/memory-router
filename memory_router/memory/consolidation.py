"""Memory consolidation — merge near-duplicate and related memories.

Periodically scans the memory store for clusters of similar memories and
merges them into a single, higher-confidence entry. This prevents memory
bloat where the same fact is restated in slightly different ways.

The algorithm:
  1. For each memory, find similar memories (Jaccard > threshold)
  2. Group overlapping sets into clusters
  3. Keep the highest-importance memory as the "anchor"
  4. Merge concepts and boost importance of the anchor
  5. Delete the duplicates

No embeddings or LLM calls required — pure keyword overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .sqlite_store import MemoryStore


@dataclass
class ConsolidationResult:
    """Summary of a consolidation run."""

    clusters_found: int
    memories_merged: int
    memories_remaining: int


def consolidate_memories(
    store: MemoryStore,
    similarity_threshold: float = 0.6,
    min_cluster_size: int = 2,
    dry_run: bool = False,
) -> ConsolidationResult:
    """Scan for near-duplicate memories and merge them.

    Args:
        store: The memory store to consolidate.
        similarity_threshold: Jaccard word-overlap threshold (0.0–1.0).
        min_cluster_size: Minimum cluster size to trigger a merge.
        dry_run: If True, report what would be merged but don't change anything.

    Returns:
        ConsolidationResult with merge statistics.
    """
    all_mems = store.list_all(limit=10_000)
    if len(all_mems) < min_cluster_size:
        return ConsolidationResult(0, 0, len(all_mems))

    # Build word sets for all memories
    word_sets: Dict[int, Set[str]] = {}
    for m in all_mems:
        if m.id is not None:
            word_sets[m.id] = set(m.content.lower().split())

    # Find similar pairs
    mem_by_id = {m.id: m for m in all_mems if m.id is not None}
    ids = list(word_sets.keys())
    similar_pairs: List[Tuple[int, int, float]] = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            ws_a, ws_b = word_sets[a], word_sets[b]
            if not ws_a or not ws_b:
                continue
            intersection = ws_a & ws_b
            union = ws_a | ws_b
            jaccard = len(intersection) / len(union)
            if jaccard >= similarity_threshold:
                similar_pairs.append((a, b, jaccard))

    if not similar_pairs:
        return ConsolidationResult(0, 0, len(all_mems))

    # Union-find to group clusters
    parent: Dict[int, int] = {mid: mid for mid in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b, _ in similar_pairs:
        union(a, b)

    # Group by root
    clusters: Dict[int, List[int]] = {}
    for mid in ids:
        root = find(mid)
        clusters.setdefault(root, []).append(mid)

    # Filter to actual clusters
    clusters = {
        root: members
        for root, members in clusters.items()
        if len(members) >= min_cluster_size
    }

    if dry_run or not clusters:
        merged = sum(len(m) - 1 for m in clusters.values())
        return ConsolidationResult(
            clusters_found=len(clusters),
            memories_merged=merged,
            memories_remaining=len(all_mems) - merged,
        )

    # Merge each cluster
    total_merged = 0
    for root, members in clusters.items():
        mems = [mem_by_id[mid] for mid in members if mid in mem_by_id]
        if len(mems) < min_cluster_size:
            continue

        # Pick the anchor: highest importance, then longest content
        mems.sort(key=lambda m: (-m.importance, -len(m.content)))
        anchor = mems[0]
        to_remove = mems[1:]

        # Merge concepts from all members into anchor
        all_concepts = set(anchor.concepts or [])
        for m in to_remove:
            all_concepts.update(m.concepts or [])

        # Boost anchor importance (capped at 1.0)
        boosted_importance = min(1.0, anchor.importance + 0.05 * len(to_remove))

        # Update anchor
        if anchor.id is not None:
            store.update_importance(anchor.id, boosted_importance)
            # Update concepts via direct SQL
            import json
            store.conn.execute(
                "UPDATE memories SET concepts = ? WHERE id = ?",
                (json.dumps(list(all_concepts)), anchor.id),
            )

        # Delete merged memories
        for m in to_remove:
            if m.id is not None:
                store.delete(m.id)
                total_merged += 1

    store.conn.commit()

    return ConsolidationResult(
        clusters_found=len(clusters),
        memories_merged=total_merged,
        memories_remaining=store.count(),
    )
