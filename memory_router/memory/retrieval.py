"""Shared memory retrieval helpers.

These helpers keep the context builder and MCP tools aligned so the same
retrieval policy is used everywhere.

v2: Adds vector search fusion, relevance thresholds, and decay reinforcement.
"""

from __future__ import annotations

from typing import List, Optional

from ..classifier import Classification
from .sqlite_store import Memory, MemoryStore

# Minimum effective score to include in results (prevents noise)
DEFAULT_RELEVANCE_THRESHOLD = 0.05


def _effective_score(mem: Memory) -> float:
    """Compute an effective relevance score combining importance and confidence."""
    return mem.importance * mem.confidence


def retrieve_relevant_memories(
    store: MemoryStore,
    classification: Classification,
    query: str,
    limit: int,
    mycelium=None,
    vector_store=None,
    query_embedding: Optional[List[float]] = None,
    touch: bool = True,
    strengthen: bool = True,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> List[Memory]:
    """Return relevant memories and optionally strengthen mycelium links.

    This keeps the search, vector fusion, graph expansion, and co-retrieval
    reinforcement logic in one place so CLI and MCP surfaces stay
    behaviorally aligned.

    Score fusion strategy:
      - FTS/keyword search provides the base candidate set
      - Vector search (if available) adds semantic matches missed by FTS
      - Mycelium spread activation adds associative graph neighbors
      - Results are merged by ID and sorted by effective_score
    """
    # 1. FTS / keyword search (primary)
    used_memories = store.search(
        task=classification.task,
        domain=classification.domain,
        concepts=classification.concepts,
        query_text=query,
        limit=limit * 2,  # over-fetch for fusion dedup
    )

    # 2. Vector search fusion (if embedding + vector store available)
    if vector_store and query_embedding:
        try:
            vector_hits = vector_store.query(query_embedding, top_k=limit)
            existing_ids = {m.id for m in used_memories}
            for hit in vector_hits:
                if hit.memory_id not in existing_ids and hit.score > relevance_threshold:
                    extra = store.get(hit.memory_id)
                    if extra:
                        used_memories.append(extra)
                        existing_ids.add(extra.id)
        except Exception:
            pass  # Vector search is best-effort

    # 3. Mycelium graph expansion
    if mycelium and used_memories:
        seed_ids = [m.id for m in used_memories if m.id is not None]
        if seed_ids:
            associated = mycelium.spread_activation(seed_ids, max_hops=2, top_k=3)
            if associated:
                existing_ids = {m.id for m in used_memories}
                for mid, _score in associated:
                    if mid not in existing_ids:
                        extra_mem = store.get(mid)
                        if extra_mem:
                            used_memories.append(extra_mem)

            if strengthen:
                all_ids = [m.id for m in used_memories if m.id is not None]
                if all_ids:
                    mycelium.strengthen_co_retrieved(all_ids)

    # 4. Relevance threshold filtering
    used_memories = [
        m for m in used_memories
        if _effective_score(m) >= relevance_threshold
    ]

    # 5. Sort by effective score (importance * confidence) descending
    used_memories.sort(key=_effective_score, reverse=True)

    # 6. Trim to requested limit
    used_memories = used_memories[:limit]

    # 7. Touch / reinforce retrieved memories
    if touch:
        for mem in used_memories:
            if mem.id is not None:
                store.touch(mem.id)

    # 8. Reinforce via decay system (if available)
    if strengthen:
        try:
            from .decay import reinforce
            for mem in used_memories:
                if mem.id is not None:
                    reinforce(store, mem.id, boost=0.05)
        except Exception:
            pass  # Reinforcement is best-effort

    return used_memories
