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
    if estimate_messages_tokens(messages) <= budget:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    chat_msgs = [m for m in messages if m.get("role") != "system"]

    # Drop oldest chat messages until we fit.
    while chat_msgs and estimate_messages_tokens(system_msgs + chat_msgs) > budget:
        chat_msgs.pop(0)

    # Last resort: truncate the final message itself.
    if chat_msgs and estimate_messages_tokens(system_msgs + chat_msgs) > budget:
        last = chat_msgs[-1]
        budget_for_last = max(200, budget - estimate_messages_tokens(system_msgs))
        max_chars = budget_for_last * 4
        last["content"] = last["content"][:max_chars]

    return system_msgs + chat_msgs
