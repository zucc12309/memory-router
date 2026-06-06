"""Tests for provider streaming support."""

from unittest.mock import MagicMock, patch

from memory_router.providers.base import BaseProvider, ProviderResult, StreamChunk


def test_stream_chunk_dataclass():
    """StreamChunk should carry text and metadata."""
    chunk = StreamChunk(text="hello", finished=False)
    assert chunk.text == "hello"
    assert chunk.finished is False
    assert chunk.input_tokens == 0
    assert chunk.output_tokens == 0

    final = StreamChunk(text="", finished=True, input_tokens=100, output_tokens=50)
    assert final.finished is True
    assert final.input_tokens == 100


def test_base_provider_stream_fallback():
    """BaseProvider.stream() should fall back to non-streaming complete()."""

    class TestProvider(BaseProvider):
        name = "test"

        def is_available(self):
            return True

        def complete(self, model, messages, **kwargs):
            return ProviderResult(
                text="full response",
                model=model,
                input_tokens=10,
                output_tokens=5,
            )

    provider = TestProvider()
    chunks = list(provider.stream("test-model", [{"role": "user", "content": "hi"}]))
    assert len(chunks) == 1
    assert chunks[0].text == "full response"
    assert chunks[0].finished is True
    assert chunks[0].input_tokens == 10


def test_ollama_stream_parsing():
    """OllamaProvider.stream() should parse NDJSON lines."""
    from memory_router.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider()

    # Mock the requests.post to return streaming NDJSON
    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_lines.return_value = [
        b'{"message":{"content":"Hello"},"done":false}',
        b'{"message":{"content":" world"},"done":false}',
        b'{"message":{"content":""},"done":true}',
    ]

    with patch("requests.post", return_value=mock_response):
        chunks = list(provider.stream("llama3", [{"role": "user", "content": "hi"}]))

    # Should have content chunks + final
    texts = [c.text for c in chunks if c.text]
    assert "Hello" in texts
    assert " world" in texts
    assert chunks[-1].finished is True


def test_stream_chunk_final_has_tokens():
    """Final StreamChunk should carry token counts."""
    final = StreamChunk(text="", finished=True, input_tokens=200, output_tokens=100)
    assert final.input_tokens == 200
    assert final.output_tokens == 100
    assert final.finished is True
