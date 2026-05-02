"""SQLite-backed memory and conversation store.

Two databases live under ~/.memory-router/:
  - memories.sqlite       — structured Memory Palace entries
  - conversations.sqlite  — chat turns for short-term recall

Both use stdlib sqlite3, so no extra dependencies. A single connection per
process is fine for a CLI; we open lazily and let SQLite handle locking.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..config import CONVERSATIONS_DB, MEMORIES_DB, ensure_dirs


# ---------- data classes ----------

@dataclass
class Memory:
    id: Optional[int] = None
    task: str = "general"
    domain: str = "general"
    concepts: List[str] = field(default_factory=list)
    content: str = ""
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    usage_count: int = 0


@dataclass
class Message:
    id: Optional[int] = None
    session_id: str = "default"
    role: str = "user"  # user | assistant | system
    content: str = ""
    created_at: float = field(default_factory=time.time)


# ---------- schema ----------

_MEMORIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    domain TEXT NOT NULL,
    concepts TEXT NOT NULL,         -- JSON array
    content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL,
    last_used REAL NOT NULL DEFAULT 0,
    usage_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_task ON memories(task);
"""

_CONVERSATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


def _connect(path: Path, schema: str) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(schema)
    conn.commit()
    return conn


# ---------- memory store ----------

class MemoryStore:
    """CRUD + simple keyword retrieval for Memory Palace entries."""

    def __init__(self, path: Path = MEMORIES_DB):
        self.conn = _connect(path, _MEMORIES_SCHEMA)

    def add(self, mem: Memory) -> int:
        cur = self.conn.execute(
            """INSERT INTO memories
               (task, domain, concepts, content, importance, created_at, last_used, usage_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mem.task,
                mem.domain,
                json.dumps(mem.concepts),
                mem.content,
                mem.importance,
                mem.created_at,
                mem.last_used,
                mem.usage_count,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def delete(self, memory_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def clear(self) -> int:
        cur = self.conn.execute("DELETE FROM memories")
        self.conn.commit()
        return cur.rowcount

    def list_all(self, limit: int = 100) -> List[Memory]:
        rows = self.conn.execute(
            "SELECT * FROM memories ORDER BY importance DESC, last_used DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def search(
        self,
        task: Optional[str] = None,
        domain: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Memory]:
        """Score by keyword overlap + importance + recency. Good enough for MVP.

        A vector store can be plugged in later (see vector_store.py) — this
        function just returns the top-N memories most likely to be relevant.
        """
        rows = self.conn.execute("SELECT * FROM memories").fetchall()
        scored = []
        concepts = [c.lower() for c in (concepts or [])]
        for r in rows:
            mem = _row_to_memory(r)
            score = mem.importance
            if task and mem.task == task:
                score += 0.5
            if domain and mem.domain == domain:
                score += 0.5
            mem_concepts = [c.lower() for c in mem.concepts]
            overlap = len(set(mem_concepts) & set(concepts))
            score += 0.3 * overlap
            # Light recency boost so frequently-used memories surface.
            if mem.last_used:
                score += min(0.2, (time.time() - mem.last_used) < 86400 and 0.2 or 0.0)
            scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def touch(self, memory_id: int) -> None:
        """Mark a memory as recently used. Called after retrieval."""
        self.conn.execute(
            "UPDATE memories SET last_used = ?, usage_count = usage_count + 1 WHERE id = ?",
            (time.time(), memory_id),
        )
        self.conn.commit()


def _row_to_memory(r: sqlite3.Row) -> Memory:
    return Memory(
        id=r["id"],
        task=r["task"],
        domain=r["domain"],
        concepts=json.loads(r["concepts"] or "[]"),
        content=r["content"],
        importance=r["importance"],
        created_at=r["created_at"],
        last_used=r["last_used"],
        usage_count=r["usage_count"],
    )


# ---------- conversation store ----------

class ConversationStore:
    """Append-only chat log; we read the tail for short-term context."""

    def __init__(self, path: Path = CONVERSATIONS_DB):
        self.conn = _connect(path, _CONVERSATIONS_SCHEMA)

    def add(self, msg: Message) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (msg.session_id, msg.role, msg.content, msg.created_at),
        )
        self.conn.commit()
        return cur.lastrowid

    def recent(self, session_id: str = "default", limit: int = 6) -> List[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        # Restore chronological order.
        rows.reverse()
        return [
            Message(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def all_for_session(self, session_id: str = "default") -> List[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [
            Message(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def clear(self, session_id: Optional[str] = None) -> int:
        if session_id:
            cur = self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        else:
            cur = self.conn.execute("DELETE FROM messages")
        self.conn.commit()
        return cur.rowcount
