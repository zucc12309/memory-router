"""Cheap, deterministic summarizer for chat history.

We avoid making an LLM call here because summarization happens on the hot path
of every query. The heuristic keeps the most recent turns verbatim and
condenses older turns into one-line role-tagged extracts. Plug in an LLM-based
summarizer later if you want richer compression.
"""

from __future__ import annotations

from typing import List

from .sqlite_store import Message


def summarize_history(messages: List[Message], keep_recent: int = 6, max_chars: int = 600) -> str:
    """Return a compact summary string for the older portion of the chat."""
    if len(messages) <= keep_recent:
        return ""
    older = messages[:-keep_recent]
    bits = []
    for m in older:
        snippet = m.content.strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        bits.append(f"{m.role}: {snippet}")
    summary = " | ".join(bits)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary
