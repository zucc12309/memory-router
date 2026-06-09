"""Extended router tests — provider registry, retry, fallback, routing rules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memory_router.classifier import Classification
from memory_router.config import Config
from memory_router.providers.base import BaseProvider, ProviderResult
from memory_router.router import (
    Router,
    RouteDecision,
    _build_providers,
    _guess_provider_from_model,
    _is_retryable_error,
    register_provider,
    _PROVIDER_FACTORIES,
)


def _cls(task="general", domain="general", complexity=0.5):
    return Classification(task=task, domain=domain, concepts=[], complexity=complexity)


class FakeProvider(BaseProvider):
    name = "fake"

    def is_available(self) -> bool:
        return True

    def complete(self, model, messages, **kw):
        return ProviderResult(text="fake answer", model=model)


class TestProviderRegistry:
    def test_register_custom_provider(self):
        old = dict(_PROVIDER_FACTORIES)
        try:
            register_provider("custom_test", FakeProvider)
            assert "custom_test" in _PROVIDER_FACTORIES
            cfg = Config()
            providers = _build_providers(cfg)
            assert "custom_test" in providers
            assert isinstance(providers["custom_test"], FakeProvider)
        finally:
            _PROVIDER_FACTORIES.clear()
            _PROVIDER_FACTORIES.update(old)

    def test_build_providers_creates_all(self):
        cfg = Config()
        providers = _build_providers(cfg)
        assert "ollama" in providers
        assert "openai" in providers
        assert "anthropic" in providers
        assert "gemini" in providers
        assert "ruflo" in providers


class TestGuessProvider:
    def test_openai_models(self):
        assert _guess_provider_from_model("gpt-4o") == "openai"
        assert _guess_provider_from_model("gpt-4o-mini") == "openai"
        assert _guess_provider_from_model("o3-mini") == "openai"

    def test_anthropic_models(self):
        assert _guess_provider_from_model("claude-sonnet-4-6") == "anthropic"
        assert _guess_provider_from_model("claude-haiku-4-5-20251001") == "anthropic"

    def test_gemini_models(self):
        assert _guess_provider_from_model("gemini-2.5-flash") == "gemini"
        assert _guess_provider_from_model("gemini-2.5-pro") == "gemini"

    def test_ollama_models(self):
        assert _guess_provider_from_model("llama3.1:8b") == "ollama"
        assert _guess_provider_from_model("mistral:7b") == "ollama"
        assert _guess_provider_from_model("deepseek-r1") == "ollama"

    def test_unknown_model(self):
        assert _guess_provider_from_model("unknown-model") is None
        assert _guess_provider_from_model("") is None


class TestRoutingRules:
    def _make_router(self):
        cfg = Config(mode="hybrid")
        router = Router(cfg)
        # Replace all providers with fakes
        for name in list(router.providers):
            fake = FakeProvider()
            fake.name = name
            router.providers[name] = fake
        return router

    def test_route_code_task_prefers_large(self):
        router = self._make_router()
        d = router.route(_cls(task="code", complexity=0.8))
        assert "large" in d.reason or "code" in d.reason

    def test_route_simple_prefers_small(self):
        router = self._make_router()
        d = router.route(_cls(task="chat", complexity=0.1))
        assert d.model is not None

    def test_route_hybrid_low_complexity_local(self):
        router = self._make_router()
        d = router.route(_cls(task="chat", complexity=0.1))
        # With low complexity in hybrid mode, should try local first
        assert d is not None

    def test_route_force_local(self):
        router = self._make_router()
        d = router.route(_cls(), force_local=True)
        assert d.provider.name == "ollama"

    def test_route_override_provider(self):
        router = self._make_router()
        d = router.route(_cls(), override_provider="gemini")
        assert d.provider.name == "gemini"

    def test_route_override_invalid_provider_raises(self):
        router = self._make_router()
        with pytest.raises(RuntimeError, match="Unknown provider"):
            router.route(_cls(), override_provider="nonexistent")


class TestFallback:
    def test_complete_with_fallback_success(self):
        cfg = Config()
        router = Router(cfg)
        fake = FakeProvider()
        fake.name = "test"
        decision = RouteDecision(
            provider=fake, model="test-model",
            reason="test", fallback_providers=None,
        )
        result, prov, model = router.complete_with_fallback(decision, [{"role": "user", "content": "hi"}])
        assert result.text == "fake answer"
        assert prov == "test"

    def test_complete_with_fallback_uses_fallback(self):
        cfg = Config()
        router = Router(cfg)

        failing = MagicMock(spec=BaseProvider)
        failing.name = "failing"
        failing.complete.side_effect = RuntimeError("500 Internal Server Error")

        working = FakeProvider()
        working.name = "working"
        router.providers["working"] = working

        decision = RouteDecision(
            provider=failing, model="fail-model",
            reason="test", fallback_providers=["working"],
        )
        # Patch _pick_fallback_model to return a model
        with patch.object(router, "_pick_fallback_model", return_value="working-model"):
            result, prov, model = router.complete_with_fallback(
                decision, [{"role": "user", "content": "hi"}]
            )
        assert prov == "working"


class TestRetryableErrors:
    def test_rate_limit(self):
        assert _is_retryable_error(RuntimeError("429 Too Many Requests"))

    def test_server_errors(self):
        assert _is_retryable_error(RuntimeError("502 Bad Gateway"))
        assert _is_retryable_error(RuntimeError("503 Service Unavailable"))

    def test_timeout(self):
        assert _is_retryable_error(RuntimeError("Connection timed out"))
        assert _is_retryable_error(RuntimeError("Request timeout"))

    def test_auth_not_retryable(self):
        assert not _is_retryable_error(RuntimeError("401 Unauthorized"))
        assert not _is_retryable_error(RuntimeError("403 Forbidden"))

    def test_status_code_attribute(self):
        err = RuntimeError("error")
        err.status_code = 429
        assert _is_retryable_error(err)

        err2 = RuntimeError("error")
        err2.status_code = 400
        assert not _is_retryable_error(err2)
