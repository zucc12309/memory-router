"""Token estimation utilities.

We avoid hard dependencies on tiktoken so the MVP runs anywhere. The estimator
uses a ~4-chars-per-token heuristic which is accurate enough for budget checks
and the savings display. Swap in tiktoken later for precision.
"""

from __future__ import annotations

from typing import Iterable


CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Rough token count using the 4-chars-per-token rule of thumb."""
    if not text:
        return 0
    # Add 1 to avoid 0 for very short non-empty strings.
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: Iterable[dict]) -> int:
    """Estimate tokens for a list of {role, content} messages."""
    total = 0
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else str(m)
        total += estimate_tokens(content) + 4  # small overhead per message
    return total


def percent_saved(full_tokens: int, sent_tokens: int) -> int:
    """Percentage of tokens saved by sending the trimmed context vs the full history."""
    if full_tokens <= 0:
        return 0
    saved = max(0, full_tokens - sent_tokens)
    return int(round(100 * saved / full_tokens))
