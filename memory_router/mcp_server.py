"""Memory Router MCP server.

Exposes the Memory Palace + prompt-optimization layer to any MCP-compatible
client (Claude Code, Cursor, Cline, Continue, Zed, etc.). The client decides
when to call these tools — Memory Router is a pure data + context layer here,
it never writes files or runs commands.

Run with:
    memory-router mcp serve

Then register with Claude Code (one-time):
    claude mcp add memory-router -- memory-router mcp serve

All tools record their token impact into stats.sqlite so cumulative savings
can be inspected with `memory-router stats` or the `stats_summary` tool.
"""

from __future__ import annotations

from typing import List, Optional

from .classifier import classify
from .config import Config, ensure_dirs, load_config
from .context_builder import build_context as _build_context
from .memory.auto_capture import capture_turn
from .memory.palace import build_palace
from .memory.sqlite_store import (
    ConversationStore,
    Memory,
    MemoryStore,
    Message,
)
from .stats import record_usage, summarize_stats, reset_stats
from .utils.tokens import percent_saved


def _create_server():
    """Lazy import so the `mcp` SDK is only required when the server runs."""
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "The `mcp` package is not installed. Install with:\n"
            '  pip install "memory-router[mcp]"'
        ) from e

    server = FastMCP("memory-router")

    # ---- Memory Palace tools ----

    @server.tool()
    def memory_search(query: str, top_k: int = 5) -> dict:
        """Search the local Memory Palace for entries relevant to the user's query.

        Use this when you need durable background context about the user —
        their stack, preferences, recurring jargon, or prior decisions —
        before answering an ambiguous prompt. Returns the top_k entries
        ranked by domain/task match, concept overlap, importance, and
        recency. Calling this updates the `last_used` and `usage_count`
        fields on retrieved memories.

        Args:
            query: The user's prompt (or any text representing intent).
            top_k: Maximum number of memories to return.

        Returns:
            classification: rule-based (task, domain, concepts, complexity)
            memories: list of {id, task, domain, concepts, content,
                              importance, usage_count}
        """
        cls = classify(query)
        store = MemoryStore()
        mems = store.search(
            task=cls.task,
            domain=cls.domain,
            concepts=cls.concepts,
            limit=top_k,
        )
        for m in mems:
            if m.id is not None:
                store.touch(m.id)
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
    ) -> dict:
        """Save a durable fact into the local Memory Palace.

        Call this when the user explicitly says to "remember", "always", or
        "never", OR when they share a stable preference, fact, or context
        that will be useful in future conversations (their stack, naming
        conventions, project context, recurring jargon). Avoid storing:
        sensitive info (keys, passwords), one-off transient details, or
        redundant content already covered by an existing memory.

        Args:
            content: A concise factual statement, ideally one sentence.
            domain: Free-form domain tag (e.g. 'software', 'finance', 'prefs').
            task: Free-form task tag (e.g. 'code', 'explain', 'reasoning').
            concepts: Keywords that boost retrieval for related queries.
            importance: 0.0–1.0; higher = retrieved more often.

        Returns: {id: int, ok: bool}
        """
        store = MemoryStore()
        memory_id = store.add(
            Memory(
                content=content.strip(),
                domain=domain,
                task=task,
                concepts=list(concepts or []),
                importance=max(0.0, min(1.0, importance)),
            )
        )
        return {"id": memory_id, "ok": True}

    @server.tool()
    def memory_list(limit: int = 20) -> dict:
        """List memories ordered by importance + recency.

        Args:
            limit: Maximum number of memories to return.
        """
        store = MemoryStore()
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
                    "usage_count": m.usage_count,
                }
                for m in mems
            ],
        }

    @server.tool()
    def memory_palace() -> dict:
        """Show all memories grouped by domain → task hierarchy."""
        store = MemoryStore()
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
        """Delete a memory by id. Returns {ok: bool}."""
        store = MemoryStore()
        return {"ok": store.delete(memory_id), "id": memory_id}

    @server.tool()
    def memory_capture(
        query: str,
        answer: str,
        domain: Optional[str] = None,
        importance: Optional[float] = None,
    ) -> dict:
        """Promote a useful completed turn into a long-term memory.

        Call this AFTER you've helped the user with something that surfaced
        knowledge worth keeping (their stack, a refactor pattern, a learned
        preference). Uses the same safety filters as auto-capture: skips
        sensitive content, prompt-injection patterns, and code-heavy turns.

        Args:
            query: The user's original prompt.
            answer: The text response (or a short summary of it).
            domain: Optional override for the inferred domain.
            importance: Optional override (0.0–1.0).

        Returns: {captured: bool, id: int|None}
        """
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
        store = MemoryStore()
        mid = capture_turn(
            query=query,
            answer=answer,
            classification=cls,
            cfg=cfg,
            store=store,
            allow_capture=True,
        )
        # Apply importance override if requested.
        if mid is not None and importance is not None:
            try:
                conn = store.conn
                conn.execute(
                    "UPDATE memories SET importance = ? WHERE id = ?",
                    (max(0.0, min(1.0, importance)), mid),
                )
                conn.commit()
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
        compressed history summary + last 5–8 turns + the query) that you
        can hand to any LLM, plus the token statistics so the user can see
        what was saved compared to a naive 'send everything' approach.

        This does NOT call an LLM. Use it when you want the optimization
        without paying for an inference call, or when you'll forward the
        messages to your own model selection.

        Args:
            query: The user's prompt.
            session_id: Conversation session id (default 'default').
            use_memory: If False, skip Memory Palace retrieval for this call.

        Returns:
            messages: list of {role, content} ready to send to an LLM
            memories_used: list of {id, content} that were injected
            naive_baseline_tokens: estimated tokens for the full history
            sent_tokens: estimated tokens actually shipped
            saved_pct: percentage saved (heuristic estimate)
            classification: {task, domain, concepts, complexity}
        """
        cfg = load_config()
        cls = classify(query)
        mem_store = MemoryStore()
        conv_store = ConversationStore()
        built = _build_context(
            query=query,
            classification=cls,
            cfg=cfg,
            mem_store=mem_store,
            conv_store=conv_store,
            use_memory=use_memory,
            session_id=session_id,
        )
        saved = percent_saved(built.full_history_tokens, built.sent_tokens)

        # Track in stats so cumulative savings show in `memory-router stats`.
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
        """Record a completed Q&A turn into the conversation log.

        Useful when the MCP client (Claude Code, Cursor, etc.) handles the
        actual LLM call and wants Memory Router to track the turn for
        future context retrieval. Optional — if you don't call this, the
        client's own history is unaffected.
        """
        store = ConversationStore()
        store.add(Message(session_id=session_id, role="user", content=query))
        store.add(Message(session_id=session_id, role="assistant", content=answer))
        return {"ok": True}

    # ---- Stats tools ----

    @server.tool()
    def stats_summary() -> dict:
        """Return cumulative token-saving stats across all CLI + MCP usage."""
        return summarize_stats().to_dict()

    @server.tool()
    def stats_reset() -> dict:
        """Reset all cumulative stats to zero. Returns rows deleted."""
        return {"deleted": reset_stats()}

    return server


def main() -> None:
    """Console entry point for `memory-router mcp serve`."""
    ensure_dirs()
    server = _create_server()
    server.run()


if __name__ == "__main__":
    main()
