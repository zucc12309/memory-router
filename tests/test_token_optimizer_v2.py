"""Extended tests for the token optimizer — priority fit and edge cases."""

from memory_router.token_optimizer import fit_to_budget


def test_priority_fit_drops_lowest_first():
    """Low-priority messages should be dropped before high-priority ones."""
    messages = [
        {"role": "system", "content": "system prompt with important context"},
        {"role": "system", "content": "low priority summary that can be dropped"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "current query"},
    ]
    priorities = [0.95, 0.3, 0.7, 0.65, 1.0]

    # Very tight budget should drop the low-priority summary
    result = fit_to_budget(messages, budget=80, priorities=priorities)

    # Current query must always survive
    assert result[-1]["content"] == "current query"

    # Low priority summary should be dropped first
    contents = [m["content"] for m in result]
    if len(result) < len(messages):
        # The 0.3-priority message should be the first to go
        assert "low priority summary" not in contents


def test_priority_fit_keeps_all_when_under_budget():
    """When everything fits, nothing should be dropped."""
    messages = [
        {"role": "system", "content": "short"},
        {"role": "user", "content": "hi"},
    ]
    priorities = [0.9, 1.0]
    result = fit_to_budget(messages, budget=1000, priorities=priorities)
    assert len(result) == 2


def test_empty_messages():
    """Empty input should return empty output."""
    result = fit_to_budget([], budget=100)
    assert result == []


def test_single_message():
    """Single message should always be kept."""
    messages = [{"role": "user", "content": "hello"}]
    result = fit_to_budget(messages, budget=100)
    assert len(result) == 1


def test_positional_fit_preserves_system_and_recent():
    """Without priorities, system messages and recent turns should be preserved."""
    messages = [
        {"role": "system", "content": "important instructions"},
        {"role": "user", "content": "old question from a while ago"},
        {"role": "assistant", "content": "old answer to old question"},
        {"role": "user", "content": "more recent question here"},
        {"role": "assistant", "content": "more recent answer here"},
        {"role": "user", "content": "latest query to answer now"},
    ]
    # No priorities → positional fit
    result = fit_to_budget(messages, budget=100)

    # Latest query should always be present
    assert result[-1]["content"] == "latest query to answer now"
    # System should survive
    assert any(m["role"] == "system" for m in result)


def test_query_always_survives_extreme_budget():
    """Even with tiny budget, the query should survive."""
    messages = [
        {"role": "system", "content": "x" * 10000},
        {"role": "user", "content": "my question"},
    ]
    result = fit_to_budget(messages, budget=20)
    assert any("my question" in m["content"] for m in result)
