"""Memory Palace — the user-facing facade over the memory stores.

A "palace" is a hierarchical view of memories grouped by domain and task,
which is what the `memory-router memory palace` command renders.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from .sqlite_store import Memory, MemoryStore


@dataclass
class PalaceNode:
    domain: str
    tasks: Dict[str, List[Memory]]


def build_palace(store: MemoryStore) -> List[PalaceNode]:
    """Group memories by (domain -> task -> [memories])."""
    grouped: Dict[str, Dict[str, List[Memory]]] = defaultdict(lambda: defaultdict(list))
    for mem in store.list_all(limit=10_000):
        grouped[mem.domain][mem.task].append(mem)
    return [PalaceNode(domain=d, tasks=dict(t)) for d, t in grouped.items()]
