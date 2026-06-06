"""Memory confidence decay and reinforcement.

Memories are living entities with a confidence score that decays over time
unless reinforced by usage. This prevents stale, outdated memories from
polluting context indefinitely and rewards memories that prove useful.

Decay model: confidence_new = confidence * exp(-decay_rate * days_since_reinforcement)
Reinforcement: bumps confidence and resets the decay clock.
Pruning: archives memories whose confidence drops below a threshold.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from .sqlite_store import MemoryStore


def apply_decay(store: MemoryStore, now: Optional[float] = None) -> int:
    """Decay confidence on all memories based on time since last use.

    Called lazily on search or periodically via CLI/cron.
    Returns count of memories updated.
    """
    now = now or time.time()
    conn = store.conn

    # Use last_used as the reinforcement signal; fall back to created_at
    # if the memory has never been used.
    cur = conn.execute(
        """UPDATE memories
           SET importance = MAX(0.01, importance * (
               CASE
                   WHEN last_used > 0
                   THEN EXP(-0.001 * ((? - last_used) / 86400.0))
                   ELSE EXP(-0.001 * ((? - created_at) / 86400.0))
               END
           ))
           WHERE importance > 0.01
             AND (
                 (last_used > 0 AND (? - last_used) > 86400)
                 OR (last_used = 0 AND (? - created_at) > 86400)
             )
        """,
        (now, now, now, now),
    )
    conn.commit()
    return cur.rowcount


def reinforce(store: MemoryStore, memory_id: int, boost: float = 0.1) -> None:
    """Reinforce a memory — increases importance and resets decay clock.

    Called when a memory is retrieved and used in context.
    """
    now = time.time()
    conn = store.conn
    conn.execute(
        """UPDATE memories
           SET importance = MIN(1.0, importance + ?),
               last_used = ?,
               usage_count = usage_count + 1
           WHERE id = ?
        """,
        (boost, now, memory_id),
    )
    conn.commit()


def prune_stale_memories(
    store: MemoryStore,
    importance_threshold: float = 0.05,
    min_age_days: float = 30.0,
) -> int:
    """Delete memories whose importance has decayed below threshold.

    Only prunes memories older than min_age_days to avoid deleting
    recently-added low-importance memories.

    Returns count of memories deleted.
    """
    now = time.time()
    cutoff = now - (min_age_days * 86400)
    conn = store.conn
    cur = conn.execute(
        """DELETE FROM memories
           WHERE importance < ?
             AND created_at < ?
        """,
        (importance_threshold, cutoff),
    )
    conn.commit()
    return cur.rowcount


def get_decay_stats(store: MemoryStore) -> dict:
    """Get statistics about memory health / decay status."""
    conn = store.conn
    row = conn.execute(
        """SELECT
               COUNT(*),
               COALESCE(AVG(importance), 0),
               COALESCE(MIN(importance), 0),
               COALESCE(MAX(importance), 0),
               SUM(CASE WHEN importance < 0.1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN importance >= 0.8 THEN 1 ELSE 0 END)
           FROM memories
        """
    ).fetchone()
    return {
        "total_memories": row[0],
        "avg_importance": round(row[1], 3),
        "min_importance": round(row[2], 3),
        "max_importance": round(row[3], 3),
        "stale_count": row[4] or 0,
        "strong_count": row[5] or 0,
    }
