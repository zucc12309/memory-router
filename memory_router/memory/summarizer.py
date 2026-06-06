"""Deterministic summarizer for chat history.

We avoid making an LLM call here because summarization happens on the hot path
of every query. The summarizer respects sentence boundaries (never cuts
mid-sentence) and uses role-tagged extracts for traceability.
"""

from __future__ import annotations

import re
from typing import List

from .sqlite_store import Message

# Sentence boundary: period, exclamation, question mark followed by space or end
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def summarize_history(
    messages: List[Message],
    keep_recent: int = 6,
    max_chars: int = 600,
) -> str:
    """Return a compact summary string for the older portion of the chat.

    Improvements over v1:
    - Respects sentence boundaries (never cuts mid-sentence)
    - Prioritizes user messages over assistant acknowledgments
    - Skips very short assistant replies (< 20 chars) that add no info
    """
    if len(messages) <= keep_recent:
        return ""

    older = messages[:-keep_recent]
    bits: List[str] = []
    budget_remaining = max_chars

    for m in older:
        # Skip trivial assistant acknowledgments
        stripped = m.content.strip()
        if m.role == "assistant" and len(stripped) < 20:
            continue

        snippet = _extract_first_sentence(stripped, max_len=120)
        entry = f"{m.role}: {snippet}"

        if budget_remaining - len(entry) - 3 < 0:
            break

        bits.append(entry)
        budget_remaining -= len(entry) + 3  # 3 for " | " separator

    return " | ".join(bits)


def _extract_first_sentence(text: str, max_len: int = 120) -> str:
    """Extract the first sentence, respecting sentence boundaries."""
    cleaned = " ".join(text.replace("\n", " ").split())
    if not cleaned:
        return ""

    # Try to find a sentence boundary within max_len
    sentences = _SENTENCE_END.split(cleaned)
    if sentences and len(sentences[0]) <= max_len:
        return sentences[0].strip()

    # No sentence boundary found — truncate at word boundary
    if len(cleaned) <= max_len:
        return cleaned

    # Find last space before max_len
    truncated = cleaned[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > max_len // 2:
        return truncated[:last_space].rstrip() + "..."
    return truncated.rstrip() + "..."
