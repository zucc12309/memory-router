from __future__ import annotations

from memory_router.token_optimizer import fit_to_budget
from memory_router.utils.tokens import estimate_messages_tokens


def test_fit_to_budget_keeps_prompt_under_budget():
    messages = [
        {"role": "system", "content": "A" * 300},
        {"role": "system", "content": "B" * 300},
        {"role": "user", "content": "C" * 300},
        {"role": "assistant", "content": "D" * 300},
        {"role": "user", "content": "E" * 300},
    ]

    trimmed = fit_to_budget(messages, budget=150)

    assert estimate_messages_tokens(trimmed) <= 150
    assert trimmed[-1]["role"] == "user"


def test_fit_to_budget_preserves_latest_context_when_possible():
    messages = [
        {"role": "system", "content": "memory block"},
        {"role": "system", "content": "summary block"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "final question"},
    ]

    trimmed = fit_to_budget(messages, budget=200)

    assert trimmed[-1]["content"] == "final question"
    assert estimate_messages_tokens(trimmed) <= 200
