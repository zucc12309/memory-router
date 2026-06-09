"""Tests for the public API exported from memory_router.__init__."""


def test_version_is_string():
    from memory_router import __version__
    assert isinstance(__version__, str)
    assert "." in __version__


def test_core_types_importable():
    """All core types should be importable from the top-level package."""
    from memory_router import (
        Classification,
        Config,
        Memory,
    )
    # Verify they're dataclasses with expected fields
    assert "task" in Classification.__dataclass_fields__
    assert "mode" in Config.__dataclass_fields__
    assert "content" in Memory.__dataclass_fields__


def test_core_functions_importable():
    """Core functions should be importable from the top-level package."""
    from memory_router import classify, build_context, load_config
    assert callable(classify)
    assert callable(build_context)
    assert callable(load_config)


def test_classify_returns_classification():
    from memory_router import classify, Classification
    result = classify("write a python function to sort a list")
    assert isinstance(result, Classification)
    assert result.task == "code"
    assert result.domain == "software"
