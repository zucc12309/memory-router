"""Token budget manager.

Given a token budget, this module trims a list of {role, content} messages
so that the most important parts survive. Uses priority scores when available
(v2 context builder attaches them), otherwise falls back to positional logic
(system messages first, recent turns preserved, oldest dropped first).
"""

from __future__ import annotations

from typing import List, Optional

from .utils.tokens import estimate_tokens, estimate_messages_tokens


def fit_to_budget(
    messages: List[dict],
    budget: int,
    priorities: Optional[List[float]] = None,
) -> List[dict]:
    """Trim messages until under `budget` tokens.

    When `priorities` is provided (same length as messages), low-priority
    messages are dropped first regardless of position.

    When `priorities` is None, the legacy behavior applies:
    system messages are preserved, oldest non-system turns are dropped first,
    and the final user message is always kept.
    """
    if not messages:
        return messages

    if priorities and len(priorities) == len(messages):
        return _priority_fit(messages, budget, priorities)

    return _positional_fit(messages, budget)


def _priority_fit(
    messages: List[dict], budget: int, priorities: List[float]
) -> List[dict]:
    """Drop lowest-priority messages first, compress mid-priority ones."""
    # Pair messages with priorities and original indices
    indexed = list(enumerate(zip(messages, priorities)))

    total = estimate_messages_tokens([m for m, _ in [ip[1] for ip in indexed]])
    if total <= budget:
        return [dict(m) for m in messages]

    # Sort by priority ascending — we'll drop from the front (lowest priority)
    indexed.sort(key=lambda x: x[1][1])

    # Always keep the last user message (query)
    query_idx = len(messages) - 1
    kept_indices = {query_idx}
    kept = [dict(messages[query_idx])]
    budget_used = estimate_messages_tokens(kept)

    # Add messages in priority order (highest first)
    for orig_idx, (msg, priority) in reversed(indexed):
        if orig_idx == query_idx:
            continue
        msg_tokens = estimate_tokens(msg.get("content", "")) + 4
        if budget_used + msg_tokens <= budget:
            kept_indices.add(orig_idx)
            budget_used += msg_tokens
        elif priority > 0.3 and budget - budget_used > 50:
            # Compress mid-priority messages to fit remaining budget
            char_budget = (budget - budget_used) * 4
            compressed = dict(msg)
            content = compressed.get("content", "")
            if len(content) > char_budget:
                compressed["content"] = content[: max(20, char_budget - 3)] + "..."
            kept_indices.add(orig_idx)
            budget_used += estimate_tokens(compressed["content"]) + 4
            # Replace the original message with the compressed version
            messages = list(messages)
            messages[orig_idx] = compressed

    # Rebuild in original order
    result = [dict(messages[i]) for i in sorted(kept_indices)]
    return result


def _positional_fit(messages: List[dict], budget: int) -> List[dict]:
    """Legacy positional trimming: keep system + recent, drop oldest."""
    trimmed = [dict(m) for m in messages]
    tail = [trimmed.pop()]

    if estimate_messages_tokens(trimmed + tail) <= budget:
        return trimmed + tail

    system_msgs = [m for m in trimmed if m.get("role") == "system"]
    chat_msgs = [m for m in trimmed if m.get("role") != "system"]

    # Drop oldest chat messages until we fit.
    while chat_msgs and estimate_messages_tokens(system_msgs + chat_msgs + tail) > budget:
        chat_msgs.pop(0)

    # Drop oldest system blocks (memory/summary notes, not fixed instructions).
    while (
        len(system_msgs) > 1
        and estimate_messages_tokens(system_msgs + chat_msgs + tail) > budget
    ):
        system_msgs.pop(0)

    # Last resort: shrink the remaining system block.
    if (
        system_msgs
        and estimate_messages_tokens(system_msgs + chat_msgs + tail) > budget
    ):
        prefix_tokens = estimate_messages_tokens(chat_msgs + tail)
        budget_for_system = max(0, budget - prefix_tokens - 4)
        # Use conservative 3 chars/token to avoid overshoot
        system_msgs[-1]["content"] = system_msgs[-1]["content"][
            : budget_for_system * 3
        ]

    # If the tail still doesn't fit, truncate it.
    if estimate_messages_tokens(system_msgs + chat_msgs + tail) > budget:
        prefix_tokens = estimate_messages_tokens(system_msgs + chat_msgs)
        budget_for_tail = max(0, budget - prefix_tokens - 4)
        tail[-1]["content"] = tail[-1]["content"][: budget_for_tail * 3]

    return system_msgs + chat_msgs + tail
