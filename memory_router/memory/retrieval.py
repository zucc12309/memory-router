"""Shared memory retrieval helpers.

These helpers keep the context builder and MCP tools aligned so the same
retrieval policy is used everywhere.
"""

from __future__ import annotations

from typing import List, Optional

from ..classifier import Classification
from .mycelium import MyceliumNetwork
from .sqlite_store import Memory, MemoryStore


def retrieve_relevant_memories(
    store: MemoryStore,
    classification: Classification,
    query: str,
    limit: int,
    mycelium: Optional[MyceliumNetwork] = None,
    touch: bool = True,
    strengthen: bool = True,
) -> List[Memory]:
    """Return relevant memories and optionally strengthen mycelium links.

    This keeps the search, graph expansion, and co-retrieval reinforcement
    logic in one place so CLI and MCP surfaces stay behaviorally aligned.
    """
    used_memories = store.search(
        task=classification.task,
        domain=classification.domain,
        concepts=classification.concepts,
        query_text=query,
        limit=limit,
    )

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

    if touch:
        for mem in used_memories:
            if mem.id is not None:
                store.touch(mem.id)

    return used_memories
