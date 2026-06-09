"""SQLite-backed memory and conversation store.

Two databases live under ~/.memory-router/:
  - memories.sqlite       — structured Memory Palace entries
  - conversations.sqlite  — chat turns for short-term recall

Both use stdlib sqlite3, so no extra dependencies. A single connection per
process is fine for a CLI; we open lazily and let SQLite handle locking.

v2 changes:
  - FTS5 full-text search for memory retrieval (replaces O(n) full-table scan)
  - memory_type column (semantic|episodic|procedural|working)
  - confidence column with decay support
  - source tracking (user|auto_capture|agent|import)
  - Cached session token count
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..config import CONVERSATIONS_DB, MEMORIES_DB, ensure_dirs
from ..utils.fs import ensure_secure_file

_log = None


def _get_log():
    global _log
    if _log is None:
        from ..utils.logging import get_logger
        _log = get_logger(__name__)
    return _log


def _sanitize_fts_term(term: str) -> str:
    """Escape FTS5 special characters by wrapping in double quotes.

    Handles code tokens like `auth.py`, `gpt-4o`, `my_func` by allowing
    alphanumeric characters plus dots, hyphens, and underscores.
    Strips FTS5 operators (*, NEAR, NOT, AND, OR) when used bare.
    """
    if not term or len(term) <= 2:
        return ""
    # Strip FTS5 wildcard/prefix operators
    cleaned = term.strip("*")
    if not cleaned:
        return ""
    # Allow alphanumeric + dots + hyphens + underscores (common in code)
    if not re.match(r'^[a-zA-Z0-9._\-]+$', cleaned):
        return ""
    # FTS5 boolean operators must not be passed unquoted
    return f'"{cleaned}"'


# ---------- data classes ----------


@dataclass
class Memory:
    id: Optional[int] = None
    task: str = "general"
    domain: str = "general"
    concepts: List[str] = field(default_factory=list)
    content: str = ""
    importance: float = 0.5
    confidence: float = 1.0
    memory_type: str = "semantic"  # semantic|episodic|procedural|working
    source: str = "user"  # user|auto_capture|agent|import
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
    confidence REAL NOT NULL DEFAULT 1.0,
    memory_type TEXT NOT NULL DEFAULT 'semantic',
    source TEXT NOT NULL DEFAULT 'user',
    created_at REAL NOT NULL,
    last_used REAL NOT NULL DEFAULT 0,
    usage_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_task ON memories(task);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, concepts,
    content=memories, content_rowid=id
);
"""

_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, concepts) VALUES (new.id, new.content, new.concepts);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, concepts) VALUES ('delete', old.id, old.content, old.concepts);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, concepts) VALUES ('delete', old.id, old.content, old.concepts);
    INSERT INTO memories_fts(rowid, content, concepts) VALUES (new.id, new.content, new.concepts);
END;
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
    ensure_secure_file(path)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(schema)
    conn.commit()
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    return conn


def _migrate_memories(conn: sqlite3.Connection) -> None:
    """Add new columns to existing databases without losing data."""
    cursor = conn.execute("PRAGMA table_info(memories)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    migrations = [
        ("confidence", "ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0"),
        ("memory_type", "ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'semantic'"),
        ("source", "ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT 'user'"),
    ]
    for col_name, sql in migrations:
        if col_name not in existing_cols:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists in some edge cases

    # Create indexes on v2 columns (must run AFTER columns exist)
    for idx_sql in [
        "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
        "CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence)",
    ]:
        try:
            conn.execute(idx_sql)
        except sqlite3.OperationalError:
            pass

    conn.commit()


def _setup_fts(conn: sqlite3.Connection) -> bool:
    """Set up FTS5 if available. Returns True if FTS is active."""
    try:
        conn.executescript(_FTS_SCHEMA)
        conn.executescript(_FTS_TRIGGERS)
        conn.commit()

        # Populate FTS from existing data (idempotent rebuild)
        try:
            conn.execute(
                """INSERT OR IGNORE INTO memories_fts(rowid, content, concepts)
                   SELECT id, content, concepts FROM memories
                   WHERE id NOT IN (SELECT rowid FROM memories_fts)"""
            )
            conn.commit()
        except sqlite3.DatabaseError:
            # FTS index corrupted — rebuild from scratch
            try:
                conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                conn.commit()
            except Exception:
                pass  # Best effort — FTS search may be degraded
        return True
    except sqlite3.OperationalError:
        # FTS5 not compiled into this SQLite build — fall back gracefully
        return False


# ---------- memory store ----------


_VALID_MEMORY_TYPES = {"semantic", "episodic", "procedural", "working"}
_VALID_SOURCES = {"user", "auto_capture", "agent", "import"}


class MemoryStore:
    """CRUD + hybrid keyword/FTS retrieval for Memory Palace entries."""

    def __init__(self, path: Path = MEMORIES_DB):
        self.conn = _connect(path, _MEMORIES_SCHEMA)
        _migrate_memories(self.conn)
        self._fts_available = _setup_fts(self.conn)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def add(self, mem: Memory) -> int:
        if mem.memory_type not in _VALID_MEMORY_TYPES:
            raise ValueError(
                f"Invalid memory_type '{mem.memory_type}'. "
                f"Valid: {', '.join(sorted(_VALID_MEMORY_TYPES))}"
            )
        if mem.source not in _VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{mem.source}'. "
                f"Valid: {', '.join(sorted(_VALID_SOURCES))}"
            )
        cur = self.conn.execute(
            """INSERT INTO memories
               (task, domain, concepts, content, importance, confidence,
                memory_type, source, created_at, last_used, usage_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mem.task,
                mem.domain,
                json.dumps(mem.concepts),
                mem.content,
                mem.importance,
                mem.confidence,
                mem.memory_type,
                mem.source,
                mem.created_at,
                mem.last_used,
                mem.usage_count,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def has_content(self, content: str) -> bool:
        """Return True if an identical memory note already exists."""
        row = self.conn.execute(
            "SELECT 1 FROM memories WHERE content = ? LIMIT 1",
            (content,),
        ).fetchone()
        return row is not None

    def find_similar(self, content: str, threshold: float = 0.7, limit: int = 5) -> List["Memory"]:
        """Find memories similar to the given content using trigram overlap.

        Returns memories whose word-level Jaccard similarity exceeds threshold.
        This is a lightweight semantic dedup check — no embeddings needed.
        """
        words = set(content.lower().split())
        if not words:
            return []
        # Use FTS5 to narrow candidates before scoring
        quoted = [_sanitize_fts_term(w) for w in list(words)[:10]]
        search_terms = " OR ".join(t for t in quoted if t)
        if not search_terms:
            return []
        try:
            rows = self.conn.execute(
                "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? LIMIT 50",
                (search_terms,),
            ).fetchall()
        except sqlite3.OperationalError:
            _get_log().warning("FTS5 query failed in find_similar", extra={"detail": search_terms})
            rows = []
        if not rows:
            return []

        ids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(ids))
        candidates = self.conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            ids,
        ).fetchall()

        results = []
        for row in candidates:
            mem = _row_to_memory(row)
            mem_words = set(mem.content.lower().split())
            if not mem_words:
                continue
            intersection = words & mem_words
            union = words | mem_words
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= threshold:
                results.append(mem)
                if len(results) >= limit:
                    break
        return results

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

    def get(self, memory_id: int) -> Optional[Memory]:
        """Retrieve a single memory by id."""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return _row_to_memory(row) if row else None

    def search(
        self,
        task: Optional[str] = None,
        domain: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        query_text: Optional[str] = None,
        memory_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 5,
    ) -> List[Memory]:
        """Hybrid search: FTS5 full-text + keyword scoring + importance + recency.

        Uses FTS5 when available for fast text matching. Falls back to the
        original keyword overlap scoring when FTS5 is not compiled in.
        """
        # Phase 1: FTS pre-filter if available and we have query text
        fts_ids = set()
        search_terms = " ".join(concepts or [])
        if query_text:
            search_terms = f"{query_text} {search_terms}"

        if self._fts_available and search_terms.strip():
            try:
                quoted = [_sanitize_fts_term(w) for w in search_terms.lower().split()]
                fts_query = " OR ".join(t for t in quoted if t)
                if fts_query:
                    fts_rows = self.conn.execute(
                        "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? LIMIT ?",
                        (fts_query, limit * 3),
                    ).fetchall()
                    fts_ids = {r[0] for r in fts_rows}
            except sqlite3.OperationalError:
                _get_log().warning("FTS5 query failed in search", extra={"detail": search_terms})

        # Phase 2: Score candidates
        if fts_ids:
            placeholders = ",".join("?" for _ in fts_ids)
            rows = self.conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})",
                list(fts_ids),
            ).fetchall()
        else:
            # Fallback: load by domain/task filter or all
            if domain and domain != "general":
                rows = self.conn.execute(
                    "SELECT * FROM memories WHERE domain = ?", (domain,)
                ).fetchall()
            elif task and task != "general":
                rows = self.conn.execute(
                    "SELECT * FROM memories WHERE task = ?", (task,)
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM memories ORDER BY importance DESC LIMIT ?",
                    (limit * 5,),
                ).fetchall()

        scored = []
        concepts_lower = [c.lower() for c in (concepts or [])]
        query_terms = _tokenize_terms(query_text) if query_text else set()

        for r in rows:
            mem = _row_to_memory(r)
            score = _score_search_candidate(
                mem,
                task=task,
                domain=domain,
                concepts_lower=concepts_lower,
                query_terms=query_terms,
                query_text=query_text,
                fts_ids=fts_ids,
                memory_type=memory_type,
                min_confidence=min_confidence,
            )
            if score is None:
                continue
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def search_by_type(
        self, memory_type: str, limit: int = 20
    ) -> List[Memory]:
        """Retrieve memories of a specific type."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE memory_type = ? ORDER BY importance DESC LIMIT ?",
            (memory_type, limit),
        ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def touch(self, memory_id: int) -> None:
        """Mark a memory as recently used. Called after retrieval."""
        self.conn.execute(
            "UPDATE memories SET last_used = ?, usage_count = usage_count + 1 WHERE id = ?",
            (time.time(), memory_id),
        )
        self.conn.commit()

    def update_importance(self, memory_id: int, importance: float) -> None:
        """Update a memory's importance score."""
        self.conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (max(0.0, min(1.0, importance)), memory_id),
        )
        self.conn.commit()

    def count(self) -> int:
        """Total number of memories."""
        row = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0


def _row_to_memory(r: sqlite3.Row) -> Memory:
    keys = r.keys()
    return Memory(
        id=r["id"],
        task=r["task"],
        domain=r["domain"],
        concepts=json.loads(r["concepts"] or "[]"),
        content=r["content"],
        importance=r["importance"],
        confidence=r["confidence"] if "confidence" in keys else 1.0,
        memory_type=r["memory_type"] if "memory_type" in keys else "semantic",
        source=r["source"] if "source" in keys else "user",
        created_at=r["created_at"],
        last_used=r["last_used"],
        usage_count=r["usage_count"],
    )


def _score_search_candidate(
    mem: Memory,
    *,
    task: Optional[str],
    domain: Optional[str],
    concepts_lower: List[str],
    query_terms: set[str],
    query_text: Optional[str],
    fts_ids: set[int],
    memory_type: Optional[str],
    min_confidence: float,
) -> Optional[float]:
    """Compute the relevance score for a single memory row."""
    if memory_type and mem.memory_type != memory_type:
        return None
    if mem.confidence < min_confidence:
        return None

    score = mem.importance * mem.confidence
    if task and mem.task == task:
        score += 0.5
    if domain and mem.domain == domain:
        score += 0.5

    mem_concepts = [c.lower() for c in mem.concepts]
    overlap = len(set(mem_concepts) & set(concepts_lower))
    score += 0.3 * overlap

    if query_terms:
        mem_terms = _tokenize_terms(" ".join([mem.content, " ".join(mem.concepts)]))
        lexical_overlap = len(query_terms & mem_terms)
        if lexical_overlap:
            score += 0.25 * lexical_overlap

        test_query = _looks_like_test_query(query_text) if query_text else False
        stack_query = _looks_like_stack_query(query_text) if query_text else False
        code_query = _looks_like_code_query(query_text) if query_text else False

        if test_query and _looks_like_test_memory(mem):
            score += 0.3
        if stack_query and _looks_like_stack_memory(mem):
            score += 0.3
        if code_query and mem.domain == "software":
            score += 0.1

    if mem.id in fts_ids:
        score += 0.4

    if mem.last_used:
        days_ago = (time.time() - mem.last_used) / 86400
        if days_ago < 1:
            score += 0.2
        elif days_ago < 7:
            score += 0.1

    score += min(0.15, mem.usage_count * 0.02)
    return score


def _tokenize_terms(text: str) -> set[str]:
    """Normalize a text blob into comparable search terms.

    We keep this intentionally lightweight: lowercase tokens, a tiny plural
    normalizer, and no external dependencies.
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-']{2,}", (text or "").lower())
    terms = set()
    for word in words:
        term = word.strip("'")
        if len(term) > 5 and term.endswith("ing"):
            term = term[:-3]
            if len(term) > 3 and term[-1:] == term[-2:-1]:
                term = term[:-1]
        if len(term) > 4 and term.endswith("ies"):
            term = term[:-3] + "y"
        elif len(term) > 4 and term.endswith("uses"):
            term = term[:-1]
        elif len(term) > 4 and term.endswith(("ses", "xes", "zes", "ches", "shes", "oes")):
            term = term[:-2]
        elif len(term) > 4 and term.endswith("es"):
            term = term[:-2]
        elif len(term) > 3 and term.endswith("s") and not term.endswith(("ss", "us", "is")):
            term = term[:-1]
        terms.add(term)
    return terms


def _looks_like_test_query(text: str) -> bool:
    terms = _tokenize_terms(text)
    return bool(terms & {"test", "pytest", "unittest", "spec", "qa"})


def _looks_like_stack_query(text: str) -> bool:
    terms = _tokenize_terms(text)
    return bool(terms & {"stack", "setup", "toolchain", "dependency", "dependencies", "version", "language", "environment"})


def _looks_like_code_query(text: str) -> bool:
    terms = _tokenize_terms(text)
    return bool(terms & {
        "code", "function", "module", "package", "script", "helper", "cli", "auth",
        "client", "endpoint", "api", "project", "repo", "implementation", "refactor",
        "bug", "fix", "debug", "patch", "test", "tests", "pytest",
    })


def _looks_like_test_memory(mem: Memory) -> bool:
    terms = _tokenize_terms(" ".join([mem.content, " ".join(mem.concepts)]))
    return bool(terms & {"test", "pytest", "unittest", "spec", "qa"})


def _looks_like_stack_memory(mem: Memory) -> bool:
    terms = _tokenize_terms(" ".join([mem.content, " ".join(mem.concepts)]))
    return bool(terms & {
        "typescript", "javascript", "python", "pnpm", "npm", "pip", "docker",
        "react", "api", "cli", "stack", "framework", "language", "runtime",
    })


# ---------- conversation store ----------


class ConversationStore:
    """Append-only chat log; we read the tail for short-term context."""

    def __init__(self, path: Path = CONVERSATIONS_DB):
        self.conn = _connect(path, _CONVERSATIONS_SCHEMA)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

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

    def count_tokens_for_session(self, session_id: str = "default") -> int:
        """Approximate total tokens without loading all message content into Python."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return max(0, row[0] // 4) if row else 0

    def session_count(self, session_id: str = "default") -> int:
        """Number of messages in a session."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def clear(self, session_id: Optional[str] = None) -> int:
        if session_id:
            cur = self.conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
        else:
            cur = self.conn.execute("DELETE FROM messages")
        self.conn.commit()
        return cur.rowcount
