"""Mycelium-inspired associative memory network.

Memories form a graph. Edges strengthen when memories are co-retrieved
(used together in the same context build). Weakly-connected memories
decay. Retrieval follows high-weight paths from query-matched nodes,
surfacing related memories that keyword search would miss.

The biological analogy: mycorrhizal networks form preferential pathways
to resources that produce the most exchange. Here, "exchange" is
co-retrieval — memories that are useful together develop strong edges.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple



_MYCELIUM_SCHEMA = """
CREATE TABLE IF NOT EXISTS mycelium_edges (
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    edge_type TEXT NOT NULL DEFAULT 'co_retrieved',
    weight REAL NOT NULL DEFAULT 1.0,
    co_retrieval_count INTEGER NOT NULL DEFAULT 0,
    last_strengthened REAL NOT NULL,
    PRIMARY KEY (source_id, target_id, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_myc_source ON mycelium_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_myc_target ON mycelium_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_myc_weight ON mycelium_edges(weight DESC);
"""


class MyceliumNetwork:
    """Graph overlay on top of MemoryStore.

    Tracks co-retrieval relationships between memories and uses spreading
    activation to surface associated memories that keyword search misses.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.executescript(_MYCELIUM_SCHEMA)
        self.conn.commit()

    def strengthen_co_retrieved(self, memory_ids: List[int], boost: float = 0.5) -> int:
        """When memories are used together, strengthen edges between them.

        This is the core mycelium behavior: pathways that carry nutrients
        (useful retrievals) grow thicker.

        Returns the number of edges updated.
        """
        if len(memory_ids) < 2:
            return 0

        now = time.time()
        edges_updated = 0

        for i, a in enumerate(memory_ids):
            for b in memory_ids[i + 1 :]:
                src, tgt = min(a, b), max(a, b)
                self.conn.execute(
                    """INSERT INTO mycelium_edges
                       (source_id, target_id, edge_type, weight, co_retrieval_count, last_strengthened)
                       VALUES (?, ?, 'co_retrieved', 1.0, 1, ?)
                       ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                           weight = MIN(10.0, weight + ?),
                           co_retrieval_count = co_retrieval_count + 1,
                           last_strengthened = ?
                    """,
                    (src, tgt, now, boost, now),
                )
                edges_updated += 1

        self.conn.commit()
        return edges_updated

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: str = "related",
        weight: float = 1.0,
    ) -> None:
        """Manually add or update a typed edge between two memories."""
        src, tgt = min(source_id, target_id), max(source_id, target_id)
        now = time.time()
        self.conn.execute(
            """INSERT INTO mycelium_edges
               (source_id, target_id, edge_type, weight, co_retrieval_count, last_strengthened)
               VALUES (?, ?, ?, ?, 0, ?)
               ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                   weight = ?,
                   last_strengthened = ?
            """,
            (src, tgt, edge_type, weight, now, weight, now),
        )
        self.conn.commit()

    def spread_activation(
        self,
        seed_ids: List[int],
        max_hops: int = 2,
        top_k: int = 5,
        min_weight: float = 0.3,
    ) -> List[Tuple[int, float]]:
        """Spreading activation from seed memories through the network.

        Like nutrients flowing through mycelium: starts at seed nodes,
        follows high-weight edges, activation decays with distance.

        Returns list of (memory_id, activation_score) sorted by score.
        """
        activation: Dict[int, float] = defaultdict(float)
        seed_set = set(seed_ids)
        for sid in seed_ids:
            activation[sid] = 1.0

        visited: Set[int] = set(seed_ids)
        frontier = list(seed_ids)

        for hop in range(max_hops):
            next_frontier: List[int] = []
            decay = 0.5 ** (hop + 1)  # activation halves each hop

            for node in frontier:
                rows = self.conn.execute(
                    """SELECT target_id AS neighbor, weight FROM mycelium_edges
                       WHERE source_id = ? AND weight >= ?
                       UNION
                       SELECT source_id AS neighbor, weight FROM mycelium_edges
                       WHERE target_id = ? AND weight >= ?
                       ORDER BY weight DESC LIMIT 10
                    """,
                    (node, min_weight, node, min_weight),
                ).fetchall()

                for neighbor_id, weight in rows:
                    spread = activation[node] * weight * decay
                    if neighbor_id not in visited:
                        activation[neighbor_id] += spread
                        next_frontier.append(neighbor_id)
                        visited.add(neighbor_id)
                    elif neighbor_id not in seed_set:
                        # Reinforce already-activated nodes
                        activation[neighbor_id] += spread * 0.5

            frontier = next_frontier

        # Return top-k activated nodes (excluding seeds)
        non_seed = [
            (mid, score) for mid, score in activation.items() if mid not in seed_set
        ]
        non_seed.sort(key=lambda x: x[1], reverse=True)
        return non_seed[:top_k]

    def get_neighbors(
        self, memory_id: int, min_weight: float = 0.1, limit: int = 10
    ) -> List[Tuple[int, float, str]]:
        """Get direct neighbors of a memory node.

        Returns list of (neighbor_id, weight, edge_type).
        """
        rows = self.conn.execute(
            """SELECT target_id, weight, edge_type FROM mycelium_edges
               WHERE source_id = ? AND weight >= ?
               UNION
               SELECT source_id, weight, edge_type FROM mycelium_edges
               WHERE target_id = ? AND weight >= ?
               ORDER BY weight DESC LIMIT ?
            """,
            (memory_id, min_weight, memory_id, min_weight, limit),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def decay_edges(self, half_life_days: float = 30.0) -> int:
        """Decay edge weights over time. Unused pathways wither.

        Returns the number of edges pruned (deleted because weight < 0.1).
        """
        now = time.time()
        self.conn.execute(
            """UPDATE mycelium_edges
               SET weight = weight * EXP(-0.693 * (? - last_strengthened) / (? * 86400))
            """,
            (now, half_life_days),
        )
        cur = self.conn.execute("DELETE FROM mycelium_edges WHERE weight < 0.1")
        self.conn.commit()
        return cur.rowcount

    def edge_count(self) -> int:
        """Total number of edges in the network."""
        row = self.conn.execute("SELECT COUNT(*) FROM mycelium_edges").fetchone()
        return row[0] if row else 0

    def remove_memory(self, memory_id: int) -> int:
        """Remove all edges involving a memory (called when a memory is deleted)."""
        cur = self.conn.execute(
            "DELETE FROM mycelium_edges WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        )
        self.conn.commit()
        return cur.rowcount

    def stats(self) -> Dict:
        """Network statistics."""
        row = self.conn.execute(
            """SELECT COUNT(*), COALESCE(AVG(weight), 0), COALESCE(MAX(weight), 0),
                      COUNT(DISTINCT source_id) + COUNT(DISTINCT target_id)
               FROM mycelium_edges"""
        ).fetchone()
        return {
            "edge_count": row[0],
            "avg_weight": round(row[1], 3),
            "max_weight": round(row[2], 3),
            "connected_nodes": row[3],
        }
