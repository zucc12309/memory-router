from __future__ import annotations

from memory_router.cli import _render_flat_prompt


def test_render_flat_prompt_keeps_recent_turns():
    prompt = _render_flat_prompt(
        [
            {"role": "system", "content": "Memory notes."},
            {"role": "assistant", "content": "Earlier answer."},
            {"role": "user", "content": "Current question."},
        ]
    )

    assert "Memory notes." in prompt
    assert "Assistant: Earlier answer." in prompt
    assert "User: Current question." in prompt
