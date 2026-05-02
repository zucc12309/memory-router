"""Vector store stub.

The MVP uses keyword scoring in sqlite_store.py because it's dependency-free
and good enough for small Memory Palaces. This module is the intended
extension point for a real vector index (FAISS, Chroma, or a hand-rolled
numpy cosine search) without changing the rest of the codebase.

Drop in your preferred backend by implementing `add` and `query`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class VectorHit:
    memory_id: int
    score: float


class VectorStore:
    """No-op default. Returns no hits, so the keyword path is used."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.enabled = False

    def add(self, memory_id: int, text: str) -> None:
        return None

    def query(self, text: str, top_k: int = 5) -> List[VectorHit]:
        return []
