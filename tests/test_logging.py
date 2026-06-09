"""Tests for structured logging."""

import json
import logging

from memory_router.utils.logging import _JSONFormatter


def test_json_formatter_output():
    """JSONFormatter should produce valid JSON with standard fields."""
    formatter = _JSONFormatter()
    record = logging.LogRecord(
        name="memory_router.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["msg"] == "test message"
    assert "ts" in parsed


def test_json_formatter_extra_fields():
    """Extra fields should be included in JSON output."""
    formatter = _JSONFormatter()
    record = logging.LogRecord(
        name="memory_router.router",
        level=logging.WARNING,
        pathname="router.py",
        lineno=42,
        msg="fallback triggered",
        args=(),
        exc_info=None,
    )
    record.provider = "openai"
    record.model = "gpt-4o"
    record.latency_ms = 500

    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["provider"] == "openai"
    assert parsed["model"] == "gpt-4o"
    assert parsed["latency_ms"] == 500


def test_json_formatter_exception():
    """Exceptions should be captured in the exception field."""
    formatter = _JSONFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="memory_router.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="something failed",
            args=(),
            exc_info=sys.exc_info(),
        )

    output = formatter.format(record)
    parsed = json.loads(output)
    assert "test error" in parsed["exception"]
