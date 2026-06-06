"""Memory Router MCP server.

Exposes the Memory Palace + prompt-optimization layer to any MCP-compatible
client (Claude Code, Cursor, Cline, Continue, Zed, etc.). The client decides
when to call these tools — Memory Router is a pure data + context layer here,
it never writes files or runs commands.

v2 changes:
  - Connection pooling: singleton stores, one connection per process
  - Rate limiting: configurable calls per minute
  - Mycelium network: associative memory retrieval
  - Working memory: session-scoped context
  - Memory decay: confidence scoring + reinforcement
  - New tools: memory_decay_stats, mycelium_stats, working_memory_*

Run with:
    memory-router mcp serve

Then register with Claude Code (one-time):
    claude mcp add memory-router -- memory-router mcp serve
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from .classifier import classify
from .config import Config, ensure_dirs, load_config
from .context_builder import build_context as _build_context
from .memory.auto_capture import capture_turn
from .memory.palace import build_palace
from .memory.retrieval import retrieve_relevant_memories
from .memory.sqlite_store import (
    ConversationStore,
    Memory,
    MemoryStore,
    Message,
)
from .stats import record_usage, summarize_stats, reset_stats
from .utils.tokens import percent_saved


# ---------- singleton stores ----------
# One connection per process. Avoids the v1 bug of opening a new
# sqlite3.connect() + DDL execution on every single tool call.

_STORES: Dict[str, object] = {}
_MYCELIUM = None
_WORKING_MEMORIES: Dict[str, object] = {}
_RATE_STATE = {"count": 0, "window_start": 0.0, "limit": 100}


def _get_stores():
    """Singleton store access."""
    if "mem" not in _STORES:
        _STORES["mem"] = MemoryStore()
        _STORES["conv"] = ConversationStore()
    return _STORES["mem"], _STORES["conv"]


def _get_mycelium():
    """Singleton mycelium network."""
    global _MYCELIUM
    if _MYCELIUM is None:
        mem_store, _ = _get_stores()
        cfg = load_config()
        if cfg.mycelium_enabled:
            from .memory.mycelium import MyceliumNetwork

            _MYCELIUM = MyceliumNetwork(mem_store.conn)
    return _MYCELIUM


def _get_working_memory(session_id: str = "default"):
    """Per-session working memory."""
    if session_id not in _WORKING_MEMORIES:
        cfg = load_config()
        from .memory.working_memory import WorkingMemory

        _WORKING_MEMORIES[session_id] = WorkingMemory(
            capacity=cfg.working_memory_capacity
        )
    return _WORKING_MEMORIES[session_id]


import re as _re

_SAFE_SESSION_RE = _re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


def _sanitize_session_id(session_id: str) -> str:
    """Validate session_id to prevent injection or abuse."""
    session_id = session_id.strip()
    if not session_id:
        return "default"
    if not _SAFE_SESSION_RE.match(session_id):
        raise ValueError(
            f"Invalid session_id: must be alphanumeric/underscore/hyphen, max 128 chars"
        )
    return session_id


def _sanitize_text(text: str, max_len: int = 100_000) -> str:
    """Truncate user-supplied text to a generous safety cap."""
    return text[:max_len].strip()


def _check_rate_limit() -> None:
    """Enforce per-minute rate limit on MCP tool calls."""
    now = time.time()
    if now - _RATE_STATE["window_start"] > 60:
        _RATE_STATE["count"] = 0
        _RATE_STATE["window_start"] = now
    _RATE_STATE["count"] += 1
    if _RATE_STATE["count"] > _RATE_STATE["limit"]:
        raise RuntimeError(
            f"Rate limit exceeded — {_RATE_STATE['limit']} calls/minute. "
            "Configure with: memory-router config set mcp_rate_limit <N>"
        )


def _create_server():
    """Lazy import so the `mcp` SDK is only required when the server runs."""
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "The `mcp` package is not installed. Install with:\n"
            '  pip install "memory-router[mcp]"'
        ) from e

    cfg = load_config()
    _RATE_STATE["limit"] = cfg.mcp_rate_limit

    server = FastMCP("memory-router")

    # ---- Memory Palace tools ----

    @server.tool()
    def memory_search(query: str, top_k: int = 5) -> dict:
        """Search the local Memory Palace for entries relevant to the user's query.

        Use this when you need durable background context about the user —
        their stack, preferences, recurring jargon, or prior decisions —
        before answering an ambiguous prompt. Returns the top_k entries
        ranked by domain/task match, concept overlap, importance, and
        recency. Uses FTS5 full-text search + mycelium associative spread.

        Args:
            query: The user's prompt (or any text representing intent).
            top_k: Maximum number of memories to return.

        Returns:
            classification: rule-based (task, domain, concepts, complexity)
            memories: list of {id, task, domain, concepts, content,
                              importance, confidence, memory_type, usage_count}
        """
        _check_rate_limit()
        query = _sanitize_text(query)
        cls = classify(query)
        store, _ = _get_stores()
        mems = retrieve_relevant_memories(
            store=store,
            classification=cls,
            query=query,
            limit=top_k,
            mycelium=_get_mycelium(),
        )

        return {
            "classification": cls.to_dict(),
            "memories": [
                {
                    "id": m.id,
                    "task": m.task,
                    "domain": m.domain,
                    "concepts": m.concepts,
                    "content": m.content,
                    "importance": m.importance,
                    "confidence": m.confidence,
                    "memory_type": m.memory_type,
                    "usage_count": m.usage_count,
                }
                for m in mems
            ],
        }

    @server.tool()
    def memory_store(
        content: str,
        domain: str = "general",
        task: str = "general",
        concepts: Optional[List[str]] = None,
        importance: float = 0.7,
        memory_type: str = "semantic",
    ) -> dict:
        """Save a durable fact into the local Memory Palace.

        Call this when the user explicitly says to "remember", "always", or
        "never", OR when they share a stable preference, fact, or context
        that will be useful in future conversations.

        Args:
            content: A concise factual statement, ideally one sentence.
            domain: Free-form domain tag (e.g. 'software', 'finance', 'prefs').
            task: Free-form task tag (e.g. 'code', 'explain', 'reasoning').
            concepts: Keywords that boost retrieval for related queries.
            importance: 0.0-1.0; higher = retrieved more often.
            memory_type: 'semantic' (facts), 'episodic' (events), 'procedural' (how-to).

        Returns: {id: int, ok: bool}
        """
        _check_rate_limit()
        content = _sanitize_text(content)
        from .memory.sqlite_store import _VALID_MEMORY_TYPES
        if memory_type not in _VALID_MEMORY_TYPES:
            return {"ok": False, "error": f"Invalid memory_type '{memory_type}'. Valid: {', '.join(sorted(_VALID_MEMORY_TYPES))}"}
        store, _ = _get_stores()
        memory_id = store.add(
            Memory(
                content=content,
                domain=domain,
                task=task,
                concepts=list(concepts or []),
                importance=max(0.0, min(1.0, importance)),
                memory_type=memory_type,
                source="user",
            )
        )
        return {"id": memory_id, "ok": True}

    @server.tool()
    def memory_list(limit: int = 20) -> dict:
        """List memories ordered by importance + recency.

        Args:
            limit: Maximum number of memories to return.
        """
        _check_rate_limit()
        store, _ = _get_stores()
        mems = store.list_all(limit=limit)
        return {
            "count": len(mems),
            "memories": [
                {
                    "id": m.id,
                    "task": m.task,
                    "domain": m.domain,
                    "concepts": m.concepts,
                    "content": m.content,
                    "importance": m.importance,
                    "confidence": m.confidence,
                    "memory_type": m.memory_type,
                    "usage_count": m.usage_count,
                }
                for m in mems
            ],
        }

    @server.tool()
    def memory_palace() -> dict:
        """Show all memories grouped by domain -> task hierarchy."""
        _check_rate_limit()
        store, _ = _get_stores()
        nodes = build_palace(store)
        return {
            "domains": [
                {
                    "domain": n.domain,
                    "tasks": {
                        task: [
                            {
                                "id": m.id,
                                "content": m.content,
                                "importance": m.importance,
                                "concepts": m.concepts,
                            }
                            for m in mems
                        ]
                        for task, mems in n.tasks.items()
                    },
                }
                for n in nodes
            ],
        }

    @server.tool()
    def memory_delete(memory_id: int) -> dict:
        """Delete a memory by id. Also removes mycelium edges. Returns {ok: bool}."""
        _check_rate_limit()
        store, _ = _get_stores()
        ok = store.delete(memory_id)
        # Clean up mycelium edges
        mycelium = _get_mycelium()
        if mycelium and ok:
            mycelium.remove_memory(memory_id)
        return {"ok": ok, "id": memory_id}

    @server.tool()
    def memory_capture(
        query: str,
        answer: str,
        domain: Optional[str] = None,
        importance: Optional[float] = None,
    ) -> dict:
        """Promote a useful completed turn into a long-term memory.

        Call this AFTER you've helped the user with something that surfaced
        knowledge worth keeping.

        Args:
            query: The user's original prompt.
            answer: The text response (or a short summary of it).
            domain: Optional override for the inferred domain.
            importance: Optional override (0.0-1.0).

        Returns: {captured: bool, id: int|None}
        """
        _check_rate_limit()
        cfg = load_config()
        cls = classify(query)
        if domain:
            from .classifier import Classification

            cls = Classification(
                task=cls.task,
                domain=domain,
                concepts=cls.concepts,
                complexity=cls.complexity,
            )
        store, _ = _get_stores()
        mid = capture_turn(
            query=query,
            answer=answer,
            classification=cls,
            cfg=cfg,
            store=store,
            allow_capture=True,
        )
        if mid is not None and importance is not None:
            try:
                store.update_importance(mid, importance)
            except Exception:
                pass
        return {"captured": mid is not None, "id": mid}

    # ---- Prompt optimization tool ----

    @server.tool()
    def build_context(
        query: str,
        session_id: str = "default",
        use_memory: bool = True,
    ) -> dict:
        """Build an optimized prompt for the query and report token savings.

        Returns the trimmed message list (top-K relevant memories +
        compressed history summary + last 5-8 turns + the query) that you
        can hand to any LLM, plus token statistics. Uses priority-scored
        context assembly, FTS5 search, and mycelium associative retrieval.

        This does NOT call an LLM.

        Args:
            query: The user's prompt.
            session_id: Conversation session id (default 'default').
            use_memory: If False, skip Memory Palace retrieval.

        Returns:
            messages, memories_used, naive_baseline_tokens, sent_tokens,
            saved_pct, classification
        """
        _check_rate_limit()
        session_id = _sanitize_session_id(session_id)
        query = _sanitize_text(query)
        cfg = load_config()
        cls = classify(query)
        mem_store, conv_store = _get_stores()
        mycelium = _get_mycelium()
        working_mem = _get_working_memory(session_id)

        built = _build_context(
            query=query,
            classification=cls,
            cfg=cfg,
            mem_store=mem_store,
            conv_store=conv_store,
            use_memory=use_memory,
            session_id=session_id,
            working_memory=working_mem,
            mycelium=mycelium,
        )
        saved = percent_saved(built.full_history_tokens, built.sent_tokens)

        record_usage(
            kind="mcp_build_context",
            naive_tokens=built.full_history_tokens,
            sent_tokens=built.sent_tokens,
            memories_used=len(built.used_memories),
        )

        return {
            "messages": built.messages,
            "memories_used": [
                {"id": m.id, "content": m.content} for m in built.used_memories
            ],
            "naive_baseline_tokens": built.full_history_tokens,
            "sent_tokens": built.sent_tokens,
            "saved_pct": saved,
            "classification": cls.to_dict(),
        }

    @server.tool()
    def log_turn(
        query: str,
        answer: str,
        session_id: str = "default",
    ) -> dict:
        """Record a completed Q&A turn into the conversation log."""
        _check_rate_limit()
        session_id = _sanitize_session_id(session_id)
        query = _sanitize_text(query)
        answer = _sanitize_text(answer, max_len=50_000)
        _, conv_store = _get_stores()
        conv_store.add(Message(session_id=session_id, role="user", content=query))
        conv_store.add(
            Message(session_id=session_id, role="assistant", content=answer)
        )

        # Advance working memory turn
        wm = _get_working_memory(session_id)
        wm.advance_turn()

        return {"ok": True}

    # ---- Working memory tools ----

    @server.tool()
    def working_memory_set(
        key: str,
        value: str,
        session_id: str = "default",
        relevance: float = 1.0,
    ) -> dict:
        """Store a value in session working memory.

        Use this for current-session context that shouldn't persist
        permanently: current file being edited, active error message,
        variable names in scope, etc.

        Args:
            key: Short identifier (e.g. 'current_file', 'error_msg').
            value: The value to store.
            session_id: Session id.
            relevance: 0.0-1.0 priority within working memory.

        Returns: {ok: bool, slots_used: int}
        """
        _check_rate_limit()
        session_id = _sanitize_session_id(session_id)
        wm = _get_working_memory(session_id)
        wm.put(key, value, relevance)
        return {"ok": True, "slots_used": wm.size}

    @server.tool()
    def working_memory_get(
        key: str, session_id: str = "default"
    ) -> dict:
        """Retrieve a value from session working memory.

        Args:
            key: The key to look up.
            session_id: Session id.

        Returns: {value: str|None, found: bool}
        """
        _check_rate_limit()
        session_id = _sanitize_session_id(session_id)
        wm = _get_working_memory(session_id)
        val = wm.get(key)
        return {"value": val, "found": val is not None}

    @server.tool()
    def working_memory_snapshot(session_id: str = "default") -> dict:
        """Show current working memory state for a session."""
        _check_rate_limit()
        session_id = _sanitize_session_id(session_id)
        wm = _get_working_memory(session_id)
        return wm.to_dict()

    # ---- Mycelium network tools ----

    @server.tool()
    def mycelium_stats() -> dict:
        """Show mycelium network statistics (edges, weights, connectivity)."""
        _check_rate_limit()
        mycelium = _get_mycelium()
        if mycelium is None:
            return {"enabled": False, "message": "Mycelium network is disabled"}
        return {"enabled": True, **mycelium.stats()}

    @server.tool()
    def mycelium_neighbors(memory_id: int, limit: int = 10) -> dict:
        """Show direct neighbors of a memory in the mycelium network.

        Args:
            memory_id: The memory to find neighbors for.
            limit: Max neighbors to return.

        Returns: {neighbors: [{id, weight, edge_type}]}
        """
        _check_rate_limit()
        mycelium = _get_mycelium()
        if mycelium is None:
            return {"enabled": False, "neighbors": []}
        neighbors = mycelium.get_neighbors(memory_id, limit=limit)
        return {
            "enabled": True,
            "neighbors": [
                {"id": n[0], "weight": round(n[1], 3), "edge_type": n[2]}
                for n in neighbors
            ],
        }

    # ---- Memory health tools ----

    @server.tool()
    def memory_decay_stats() -> dict:
        """Show memory health statistics (confidence distribution, stale count)."""
        _check_rate_limit()
        store, _ = _get_stores()
        from .memory.decay import get_decay_stats

        return get_decay_stats(store)

    @server.tool()
    def memory_prune(
        importance_threshold: float = 0.05,
        min_age_days: float = 30.0,
    ) -> dict:
        """Prune memories whose importance has decayed below threshold.

        Only prunes memories older than min_age_days.

        Args:
            importance_threshold: Delete memories below this importance.
            min_age_days: Only prune memories older than this.

        Returns: {pruned: int}
        """
        _check_rate_limit()
        store, _ = _get_stores()
        from .memory.decay import prune_stale_memories

        n = prune_stale_memories(store, importance_threshold, min_age_days)
        return {"pruned": n}

    @server.tool()
    def memory_consolidate(
        similarity_threshold: float = 0.6,
        dry_run: bool = True,
    ) -> dict:
        """Find and merge near-duplicate memories.

        Scans for memories with high word overlap and merges them into
        a single higher-confidence entry. Use dry_run=True first to
        preview what would be merged.

        Args:
            similarity_threshold: Jaccard overlap threshold (0.0-1.0).
            dry_run: If True, report without changing anything.

        Returns: {clusters_found, memories_merged, memories_remaining}
        """
        _check_rate_limit()
        store, _ = _get_stores()
        from .memory.consolidation import consolidate_memories

        result = consolidate_memories(store, similarity_threshold, dry_run=dry_run)
        return {
            "clusters_found": result.clusters_found,
            "memories_merged": result.memories_merged,
            "memories_remaining": result.memories_remaining,
            "dry_run": dry_run,
        }

    @server.tool()
    def memory_find_similar(content: str, threshold: float = 0.7) -> dict:
        """Find memories similar to given text.

        Useful for checking near-duplicates before storing a new memory.

        Args:
            content: Text to find similar memories for.
            threshold: Similarity threshold (0.0-1.0).

        Returns: {similar: [{id, content, similarity}]}
        """
        _check_rate_limit()
        content = _sanitize_text(content)
        store, _ = _get_stores()
        similar = store.find_similar(content, threshold=threshold)
        return {
            "similar": [
                {"id": m.id, "content": m.content, "importance": m.importance}
                for m in similar
            ]
        }

    # ---- Stats tools ----

    @server.tool()
    def stats_summary() -> dict:
        """Return cumulative token-saving stats across all CLI + MCP usage."""
        _check_rate_limit()
        return summarize_stats().to_dict()

    @server.tool()
    def stats_reset() -> dict:
        """Reset all cumulative stats to zero. Returns rows deleted."""
        _check_rate_limit()
        return {"deleted": reset_stats()}

    # ---- Health check tool ----

    @server.tool()
    def health_check() -> dict:
        """Run system health checks and return structured status report.

        Checks: config, memory store, FTS5, providers, tiktoken, encryption,
        memory health. Returns overall status (ok/degraded/unhealthy) and
        per-check details.
        """
        _check_rate_limit()
        from .health import check_health

        return check_health().to_dict()

    return server


def main() -> None:
    """Console entry point for `memory-router mcp serve`."""
    ensure_dirs()
    server = _create_server()
    server.run()


if __name__ == "__main__":
    main()
