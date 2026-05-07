"""Token budget manager.

Given a token budget, this module trims a list of {role, content} messages
so that the most important parts (system prompt + recent turns) survive.
"""

from __future__ import annotations

from typing import List

from .utils.tokens import estimate_tokens, estimate_messages_tokens


def fit_to_budget(messages: List[dict], budget: int) -> List[dict]:
    """Trim oldest non-system messages until under `budget` tokens.

    System messages are preserved (they hold instructions and memory context).
    The most recent user/assistant turns are preserved; older turns are dropped.
    """
    if not messages:
        return messages

    trimmed = [dict(m) for m in messages]
    tail = [trimmed.pop()]

    if estimate_messages_tokens(trimmed + tail) <= budget:
        return trimmed + tail

    system_msgs = [m for m in trimmed if m.get("role") == "system"]
    chat_msgs = [m for m in trimmed if m.get("role") != "system"]

    # Drop oldest chat messages until we fit around the final user turn.
    while chat_msgs and estimate_messages_tokens(system_msgs + chat_msgs + tail) > budget:
        chat_msgs.pop(0)

    # If the generated system notes are still too large, drop the oldest
    # generated system blocks first. In this project those are memory and
    # summary notes, not fixed model instructions.
    while len(system_msgs) > 1 and estimate_messages_tokens(system_msgs + chat_msgs + tail) > budget:
        system_msgs.pop(0)

    # Last resort: shrink the remaining system block, but keep the tail.
    if system_msgs and estimate_messages_tokens(system_msgs + chat_msgs + tail) > budget:
        prefix_tokens = estimate_messages_tokens(chat_msgs + tail)
        budget_for_system = max(0, budget - prefix_tokens - 4)
        system_msgs[-1]["content"] = system_msgs[-1]["content"][: budget_for_system * 4]

    # If the tail still doesn't fit, truncate it as the very last step.
    if estimate_messages_tokens(system_msgs + chat_msgs + tail) > budget:
        prefix_tokens = estimate_messages_tokens(system_msgs + chat_msgs)
        budget_for_tail = max(0, budget - prefix_tokens - 4)
        tail[-1]["content"] = tail[-1]["content"][: budget_for_tail * 4]

    return system_msgs + chat_msgs + tail
