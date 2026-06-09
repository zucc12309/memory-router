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


def test_fit_empty_messages():
    assert fit_to_budget([], 1000) == []


def test_fit_within_budget_unchanged():
    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result = fit_to_budget(msgs, 10000)
    assert len(result) == 2


def test_priority_fit_keeps_query():
    msgs = [
        {"role": "user", "content": "big context " * 500},
        {"role": "user", "content": "my question"},
    ]
    priorities = [0.9, 1.0]
    result = fit_to_budget(msgs, 50, priorities)
    assert any("my question" in m["content"] for m in result)


def test_priority_drops_low_first():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "low priority " * 100},
        {"role": "user", "content": "high priority " * 10},
        {"role": "user", "content": "query"},
    ]
    priorities = [0.9, 0.1, 0.8, 1.0]
    result = fit_to_budget(msgs, 200, priorities)
    assert "query" in result[-1]["content"]


def test_positional_keeps_system():
    msgs = [
        {"role": "system", "content": "important"},
        {"role": "user", "content": "old " * 100},
        {"role": "user", "content": "new question"},
    ]
    result = fit_to_budget(msgs, 100)
    assert result[0]["role"] == "system"


def test_no_priorities_uses_positional():
    msgs = [{"role": "user", "content": "q"}]
    result = fit_to_budget(msgs, 10000, priorities=None)
    assert len(result) == 1
