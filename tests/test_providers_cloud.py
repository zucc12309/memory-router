"""Mock tests for cloud providers: OpenAI, Anthropic, Gemini.

Each provider uses lazy SDK imports and keychain auth. We mock the SDK
classes and get_secret to test complete(), stream(), is_available(),
and _ensure_client() without real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    def _make(self, api_key="sk-test"):
        with patch(
            "memory_router.providers.openai_provider.get_secret",
            return_value=api_key,
        ):
            from memory_router.providers.openai_provider import OpenAIProvider
            return OpenAIProvider()

    def test_is_available_with_key_and_sdk(self):
        prov = self._make("sk-test")
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            assert prov.is_available() is True

    def test_not_available_without_key(self):
        prov = self._make(None)
        assert prov.is_available() is False

    def test_not_available_without_sdk(self):
        prov = self._make("sk-test")
        import sys
        # Temporarily remove openai if present
        saved = sys.modules.get("openai")
        sys.modules["openai"] = None
        try:
            # ImportError when module is None in sys.modules
            assert prov.is_available() is False
        finally:
            if saved is not None:
                sys.modules["openai"] = saved
            else:
                sys.modules.pop("openai", None)

    def test_ensure_client_no_key_raises(self):
        prov = self._make(None)
        with pytest.raises(RuntimeError, match="No OpenAI API key"):
            # Provide openai in sys.modules so the import doesn't fail first
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                prov._ensure_client()

    def test_ensure_client_no_sdk_raises(self):
        prov = self._make("sk-test")
        import sys
        saved = sys.modules.get("openai")
        sys.modules["openai"] = None
        try:
            with pytest.raises(RuntimeError, match="openai package not installed"):
                prov._ensure_client()
        finally:
            if saved is not None:
                sys.modules["openai"] = saved
            else:
                sys.modules.pop("openai", None)

    def test_complete_returns_provider_result(self):
        prov = self._make("sk-test")
        mock_client = MagicMock()

        # Mock response
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 8
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_resp

        prov._client = mock_client
        msgs = [{"role": "user", "content": "hi"}]
        result = prov.complete("gpt-4o", msgs)

        assert result.text == "Hello world"
        assert result.model == "gpt-4o"
        assert result.input_tokens == 15
        assert result.output_tokens == 8

    def test_complete_fallback_token_estimate(self):
        """When usage is None, should fall back to token estimation."""
        prov = self._make("sk-test")
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "response text"
        mock_resp = MagicMock(spec=[])  # No 'usage' attribute
        mock_resp.choices = [mock_choice]
        type(mock_resp).usage = PropertyMock(return_value=None)
        mock_client.chat.completions.create.return_value = mock_resp

        prov._client = mock_client
        result = prov.complete("gpt-4o-mini", [{"role": "user", "content": "test"}])
        assert result.text == "response text"
        assert result.input_tokens > 0
        assert result.output_tokens > 0

    def test_stream_yields_chunks(self):
        prov = self._make("sk-test")
        mock_client = MagicMock()

        # Build streaming chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hel"
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "lo"
        chunk3 = MagicMock()
        chunk3.choices = []  # Empty choices at end

        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])
        prov._client = mock_client

        chunks = list(prov.stream("gpt-4o", [{"role": "user", "content": "hi"}]))
        # "Hel" chunk, "lo" chunk, final summary chunk
        assert len(chunks) == 3
        assert chunks[0].text == "Hel"
        assert chunks[0].finished is False
        assert chunks[1].text == "lo"
        assert chunks[-1].finished is True

    def test_ensure_client_caches(self):
        prov = self._make("sk-test")
        sentinel = object()
        prov._client = sentinel
        assert prov._ensure_client() is sentinel


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    def _make(self, api_key="sk-ant-test"):
        with patch(
            "memory_router.providers.anthropic_provider.get_secret",
            return_value=api_key,
        ):
            from memory_router.providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider()

    def test_is_available_with_key_and_sdk(self):
        prov = self._make("sk-ant-test")
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            assert prov.is_available() is True

    def test_not_available_without_key(self):
        prov = self._make(None)
        assert prov.is_available() is False

    def test_ensure_client_no_key_raises(self):
        prov = self._make(None)
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            with pytest.raises(RuntimeError, match="No Anthropic API key"):
                prov._ensure_client()

    def test_ensure_client_no_sdk_raises(self):
        prov = self._make("sk-ant-test")
        import sys
        saved = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None
        try:
            with pytest.raises(RuntimeError, match="anthropic package not installed"):
                prov._ensure_client()
        finally:
            if saved is not None:
                sys.modules["anthropic"] = saved
            else:
                sys.modules.pop("anthropic", None)

    def test_complete_with_system_message(self):
        prov = self._make("sk-ant-test")
        mock_client = MagicMock()

        # Mock response
        mock_block = MagicMock()
        mock_block.text = "Anthropic says hello"
        mock_usage = MagicMock()
        mock_usage.input_tokens = 20
        mock_usage.output_tokens = 10
        mock_resp = MagicMock()
        mock_resp.content = [mock_block]
        mock_resp.usage = mock_usage
        mock_client.messages.create.return_value = mock_resp

        prov._client = mock_client
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = prov.complete("claude-sonnet-4-20250514", msgs)

        assert result.text == "Anthropic says hello"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.input_tokens == 20
        assert result.output_tokens == 10

        # Verify system was split out
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "Be helpful."
        assert len(call_kwargs.kwargs["messages"]) == 1

    def test_complete_without_system(self):
        prov = self._make("sk-ant-test")
        mock_client = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Response"
        mock_resp = MagicMock()
        mock_resp.content = [mock_block]
        mock_resp.usage = MagicMock(input_tokens=5, output_tokens=3)
        mock_client.messages.create.return_value = mock_resp

        prov._client = mock_client
        result = prov.complete("claude-sonnet-4-20250514", [{"role": "user", "content": "hi"}])
        assert result.text == "Response"
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["system"] == ""

    def test_stream_yields_chunks(self):
        prov = self._make("sk-ant-test")
        mock_client = MagicMock()

        # Mock streaming context manager
        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["Hel", "lo", " world"])
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_client.messages.stream.return_value = mock_stream

        prov._client = mock_client
        chunks = list(prov.stream("claude-sonnet-4-20250514", [{"role": "user", "content": "hi"}]))

        # 3 text chunks + 1 final
        assert len(chunks) == 4
        assert chunks[0].text == "Hel"
        assert chunks[0].finished is False
        assert chunks[-1].finished is True
        assert chunks[-1].input_tokens > 0

    def test_ensure_client_caches(self):
        prov = self._make("sk-ant-test")
        sentinel = object()
        prov._client = sentinel
        assert prov._ensure_client() is sentinel

    def test_complete_max_tokens_kwarg(self):
        prov = self._make("sk-ant-test")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="ok")]
        mock_resp.usage = MagicMock(input_tokens=5, output_tokens=2)
        mock_client.messages.create.return_value = mock_resp

        prov._client = mock_client
        prov.complete("claude-sonnet-4-20250514", [{"role": "user", "content": "hi"}], max_tokens=2048)
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    def _make(self, api_key="gem-test"):
        with patch(
            "memory_router.providers.gemini_provider.get_secret",
            return_value=api_key,
        ):
            from memory_router.providers.gemini_provider import GeminiProvider
            return GeminiProvider()

    def test_is_available_with_key_and_sdk(self):
        prov = self._make("gem-test")
        mock_google = MagicMock()
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_google.genai}):
            assert prov.is_available() is True

    def test_not_available_without_key(self):
        prov = self._make(None)
        assert prov.is_available() is False

    def test_ensure_client_no_key_raises(self):
        prov = self._make(None)
        mock_google = MagicMock()
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_google.genai}):
            with pytest.raises(RuntimeError, match="No Gemini API key"):
                prov._ensure_client()

    def test_ensure_client_no_sdk_raises(self):
        prov = self._make("gem-test")
        import sys
        saved_google = sys.modules.get("google")
        saved_genai = sys.modules.get("google.genai")
        sys.modules["google"] = None
        sys.modules["google.genai"] = None
        try:
            with pytest.raises(RuntimeError, match="google-genai package not installed"):
                prov._ensure_client()
        finally:
            if saved_google is not None:
                sys.modules["google"] = saved_google
            else:
                sys.modules.pop("google", None)
            if saved_genai is not None:
                sys.modules["google.genai"] = saved_genai
            else:
                sys.modules.pop("google.genai", None)

    def test_prepare_messages_splits_system(self):
        prov = self._make("gem-test")
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How?"},
        ]
        system_parts, history, latest = prov._prepare_messages(msgs)
        assert system_parts == ["Be helpful."]
        assert len(history) == 2  # Hello + Hi (all but last chat msg)
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "model"  # assistant -> model
        assert latest == "How?"

    def test_prepare_messages_no_system(self):
        prov = self._make("gem-test")
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        system_parts, history, latest = prov._prepare_messages(msgs)
        assert system_parts == []
        assert latest == "World"

    def test_make_config_with_system(self):
        prov = self._make("gem-test")
        # _make_config imports google.genai.types directly, so we test
        # that it returns a non-None config when system_parts are provided
        cfg = prov._make_config(["Be helpful.", "Be concise."])
        assert cfg is not None

    def test_make_config_without_system(self):
        prov = self._make("gem-test")
        cfg = prov._make_config([])
        assert cfg is None

    def test_complete_returns_result(self):
        prov = self._make("gem-test")
        mock_client = MagicMock()
        mock_types = MagicMock()

        # Mock chat session
        mock_chat = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Gemini says hello"
        mock_resp.usage_metadata = MagicMock(prompt_token_count=12, candidates_token_count=6)
        mock_chat.send_message.return_value = mock_resp
        mock_client.chats.create.return_value = mock_chat

        prov._client = mock_client

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": MagicMock(),
            "google.genai.types": mock_types,
        }):
            result = prov.complete("gemini-2.0-flash", [
                {"role": "user", "content": "hello"},
            ])

        assert result.text == "Gemini says hello"
        assert result.model == "gemini-2.0-flash"
        assert result.input_tokens == 12
        assert result.output_tokens == 6

    def test_stream_yields_chunks(self):
        prov = self._make("gem-test")
        mock_client = MagicMock()
        mock_types = MagicMock()

        # Mock stream chunks
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"
        mock_client.models.generate_content_stream.return_value = iter([chunk1, chunk2])

        prov._client = mock_client

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": MagicMock(),
            "google.genai.types": mock_types,
        }):
            chunks = list(prov.stream("gemini-2.0-flash", [
                {"role": "user", "content": "hi"},
            ]))

        # 2 content chunks + 1 final
        assert len(chunks) == 3
        assert chunks[0].text == "Hello"
        assert chunks[0].finished is False
        assert chunks[-1].finished is True

    def test_stream_fallback_on_non_auth_error(self):
        """Non-auth stream errors should fall back to complete()."""
        prov = self._make("gem-test")
        mock_client = MagicMock()
        mock_types = MagicMock()

        # Stream raises a network error
        mock_client.models.generate_content_stream.side_effect = ConnectionError("network down")

        # Complete fallback
        mock_chat = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "fallback response"
        mock_resp.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=3)
        mock_chat.send_message.return_value = mock_resp
        mock_client.chats.create.return_value = mock_chat

        prov._client = mock_client

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": MagicMock(),
            "google.genai.types": mock_types,
        }):
            chunks = list(prov.stream("gemini-2.0-flash", [
                {"role": "user", "content": "hi"},
            ]))

        assert len(chunks) == 1
        assert chunks[0].text == "fallback response"
        assert chunks[0].finished is True

    def test_stream_reraises_auth_error(self):
        """Auth-related stream errors should not fall back."""
        prov = self._make("gem-test")
        mock_client = MagicMock()
        mock_types = MagicMock()

        mock_client.models.generate_content_stream.side_effect = Exception("API key invalid, permission denied")
        prov._client = mock_client

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": MagicMock(),
            "google.genai.types": mock_types,
        }):
            with pytest.raises(Exception, match="permission denied"):
                list(prov.stream("gemini-2.0-flash", [{"role": "user", "content": "hi"}]))

    def test_ensure_client_caches(self):
        prov = self._make("gem-test")
        sentinel = object()
        prov._client = sentinel
        assert prov._ensure_client() is sentinel
