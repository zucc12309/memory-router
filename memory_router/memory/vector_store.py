"""Vector store with progressive backend support.

Backends (in preference order):
  1. sqlite-vec (fast, no extra deps beyond sqlite)
  2. numpy cosine similarity (good enough for <10K memories)
  3. No-op fallback (keyword path is used)

The store maps memory IDs to embedding vectors and supports top-K
similarity search. Embeddings are generated externally and passed in.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from typing import List, Optional



@dataclass
class VectorHit:
    memory_id: int
    score: float


class VectorStore:
    """Vector similarity search backed by SQLite (numpy cosine when available)."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None, dim: int = 384):
        self.dim = dim
        self.enabled = False
        self._numpy = None

        if conn is not None:
            self.conn = conn
        else:
            self.conn = None

        self._try_init()

    def _try_init(self) -> None:
        """Set up the embeddings table and check for numpy."""
        try:
            import numpy as np

            self._numpy = np
        except ImportError:
            pass

        if self.conn is None:
            return

        try:
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    model TEXT DEFAULT 'unknown',
                    created_at REAL NOT NULL DEFAULT 0
                )"""
            )
            self.conn.commit()
            self.enabled = True
        except Exception:
            self.enabled = False

    def add(self, memory_id: int, embedding: List[float], model: str = "unknown") -> None:
        """Store an embedding vector for a memory ID."""
        if not self.enabled:
            return
        import time

        blob = self._to_blob(embedding)
        self.conn.execute(
            """INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding, model, created_at)
               VALUES (?, ?, ?, ?)""",
            (memory_id, blob, model, time.time()),
        )
        self.conn.commit()

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[VectorHit]:
        """Find top-K most similar memories by cosine similarity.

        Uses numpy when available for vectorized computation.
        Falls back to pure-Python dot product otherwise.
        """
        if not self.enabled:
            return []

        rows = self.conn.execute(
            "SELECT memory_id, embedding FROM memory_embeddings"
        ).fetchall()

        if not rows:
            return []

        if self._numpy is not None:
            return self._query_numpy(query_embedding, rows, top_k)

        return self._query_python(query_embedding, rows, top_k)

    def remove(self, memory_id: int) -> None:
        """Remove an embedding."""
        if not self.enabled:
            return
        self.conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
        )
        self.conn.commit()

    def count(self) -> int:
        """Number of stored embeddings."""
        if not self.enabled:
            return 0
        row = self.conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings"
        ).fetchone()
        return row[0] if row else 0

    def _query_numpy(
        self, query_embedding: List[float], rows, top_k: int
    ) -> List[VectorHit]:
        """Vectorized cosine similarity using numpy."""
        np = self._numpy
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        ids = []
        embeddings = []
        for row in rows:
            ids.append(row[0])
            embeddings.append(self._from_blob(row[1]))

        mat = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        scores = mat @ q / (norms * q_norm)

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            VectorHit(memory_id=ids[i], score=float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def _query_python(
        self, query_embedding: List[float], rows, top_k: int
    ) -> List[VectorHit]:
        """Pure-Python cosine similarity fallback."""
        q = query_embedding
        q_norm = sum(x * x for x in q) ** 0.5
        if q_norm == 0:
            return []

        scored = []
        for row in rows:
            mid = row[0]
            emb = self._from_blob(row[1])
            dot = sum(a * b for a, b in zip(q, emb))
            e_norm = sum(x * x for x in emb) ** 0.5
            if e_norm == 0:
                continue
            score = dot / (q_norm * e_norm)
            scored.append(VectorHit(memory_id=mid, score=score))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def _to_blob(self, embedding: List[float]) -> bytes:
        """Pack float list into a compact binary blob."""
        return struct.pack(f"{len(embedding)}f", *embedding)

    def _from_blob(self, blob: bytes) -> List[float]:
        """Unpack binary blob into float list."""
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))
