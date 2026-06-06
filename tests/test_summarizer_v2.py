"""Tests for the improved sentence-boundary summarizer."""

from memory_router.memory.summarizer import summarize_history, _extract_first_sentence
from memory_router.memory.sqlite_store import Message


def test_sentence_boundary_extraction():
    text = "This is the first sentence. This is the second one."
    result = _extract_first_sentence(text, max_len=120)
    assert result == "This is the first sentence."


def test_long_text_word_boundary():
    text = "A " * 100  # 200 chars of "A A A A..."
    result = _extract_first_sentence(text, max_len=50)
    assert result.endswith("...")
    assert len(result) <= 53  # 50 + "..."


def test_summarize_skips_short_assistant_replies():
    messages = [
        Message(role="user", content="Explain bond convexity"),
        Message(role="assistant", content="Ok."),
        Message(role="user", content="What about duration?"),
        Message(role="assistant", content="Duration measures the sensitivity of bond prices to interest rate changes. It's a key metric."),
        Message(role="user", content="recent1"),
        Message(role="assistant", content="recent2"),
        Message(role="user", content="recent3"),
        Message(role="assistant", content="recent4"),
        Message(role="user", content="recent5"),
        Message(role="assistant", content="recent6"),
        Message(role="user", content="recent7"),
        Message(role="assistant", content="recent8"),
    ]

    summary = summarize_history(messages, keep_recent=6)
    # "Ok." should be skipped (< 20 chars)
    assert "Ok." not in summary
    # Substantive content should be present
    assert "user:" in summary.lower() or "assistant:" in summary.lower()


def test_empty_history():
    assert summarize_history([], keep_recent=6) == ""


def test_short_history():
    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    assert summarize_history(messages, keep_recent=6) == ""
