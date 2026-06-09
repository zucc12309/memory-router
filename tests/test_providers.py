"""Tests for provider implementations — mocked HTTP, availability, streaming."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from memory_router.providers.base import BaseProvider, ProviderResult, StreamChunk
from memory_router.providers.ollama_provider import OllamaProvider
from memory_router.providers.ruflo_provider import RufloProvider


class TestBaseProvider:
    def test_split_system_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "system", "content": "Be concise."},
        ]
        system_text, chat = BaseProvider.split_system_messages(msgs)
        assert "You are helpful." in system_text
        assert "Be concise." in system_text
        assert len(chat) == 2
        assert chat[0]["role"] == "user"

    def test_split_no_system(self):
        msgs = [{"role": "user", "content": "Hello"}]
        system_text, chat = BaseProvider.split_system_messages(msgs)
        assert system_text == ""
        assert len(chat) == 1

    def test_default_stream_falls_back_to_complete(self):
        """Default stream() should call complete() and yield one chunk."""

        class SimpleProvider(BaseProvider):
            name = "simple"

            def is_available(self):
                return True

            def complete(self, model, messages, **kw):
                return ProviderResult(text="result", model=model,
                                      input_tokens=10, output_tokens=5)

        prov = SimpleProvider()
        chunks = list(prov.stream("model", [{"role": "user", "content": "hi"}]))
        assert len(chunks) == 1
        assert chunks[0].finished is True
        assert chunks[0].text == "result"
        assert chunks[0].input_tokens == 10

    def test_provider_result_fields(self):
        r = ProviderResult(
            text="hello", model="test",
            latency_ms=100, finish_reason="stop",
            retryable=False, request_id="abc123",
        )
        assert r.latency_ms == 100
        assert r.request_id == "abc123"


class TestOllamaProvider:
    def test_is_available_true(self):
        prov = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.get", return_value=mock_resp):
            assert prov.is_available() is True

    def test_is_available_false_on_error(self):
        prov = OllamaProvider()
        with patch("requests.get", side_effect=ConnectionError):
            assert prov.is_available() is False

    def test_complete(self):
        prov = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Hello world"},
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock_resp):
            result = prov.complete("llama3:8b", [{"role": "user", "content": "hi"}])
        assert result.text == "Hello world"
        assert result.model == "llama3:8b"


class TestRufloProvider:
    def test_not_available_without_package(self):
        prov = RufloProvider()
        with patch.dict("sys.modules", {"ruflo": None}):
            assert prov.is_available() is False

    def test_complete_raises_without_package(self):
        prov = RufloProvider()
        import pytest
        with pytest.raises(RuntimeError, match="ruflo not installed"):
            prov.complete("test", [{"role": "user", "content": "hi"}])


class TestStreamChunk:
    def test_defaults(self):
        chunk = StreamChunk(text="hello")
        assert chunk.finished is False
        assert chunk.input_tokens == 0
        assert chunk.output_tokens == 0
