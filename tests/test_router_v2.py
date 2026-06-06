"""Tests for the upgraded router with fallback routing."""

from memory_router.classifier import classify
from memory_router.config import Config
from memory_router.router import Router, _guess_provider_from_model


class _StubProvider:
    def __init__(self, name, available=True, fail_on_complete=False):
        self.name = name
        self._available = available
        self._fail = fail_on_complete

    def is_available(self):
        return self._available

    def complete(self, model, messages, **kwargs):
        if self._fail:
            raise RuntimeError(f"{self.name} failed")
        from memory_router.providers.base import ProviderResult
        return ProviderResult(text="ok", model=model, input_tokens=10, output_tokens=5)


def test_fallback_on_failure():
    cfg = Config(mode="api")
    router = Router(cfg)

    # Primary fails, fallback succeeds
    primary = _StubProvider("anthropic", available=True, fail_on_complete=True)
    fallback = _StubProvider("gemini", available=True, fail_on_complete=False)

    from memory_router.router import RouteDecision
    decision = RouteDecision(
        provider=primary,
        model="claude-sonnet-4-6",
        reason="test",
        fallback_providers=["gemini"],
    )
    router.providers["gemini"] = fallback

    result, used_provider = router.complete_with_fallback(decision, [{"role": "user", "content": "hi"}])
    assert result.text == "ok"
    assert used_provider == "gemini"


def test_no_fallback_raises():
    cfg = Config(mode="api")
    router = Router(cfg)

    primary = _StubProvider("anthropic", available=True, fail_on_complete=True)
    decision = router.__class__.__mro__  # Just need RouteDecision
    from memory_router.router import RouteDecision
    decision = RouteDecision(
        provider=primary,
        model="test",
        reason="test",
        fallback_providers=None,
    )

    import pytest
    with pytest.raises(RuntimeError):
        router.complete_with_fallback(decision, [{"role": "user", "content": "hi"}])


def test_guess_provider_new_models():
    assert _guess_provider_from_model("o3-mini") == "openai"
    assert _guess_provider_from_model("o4-mini") == "openai"
    assert _guess_provider_from_model("deepseek-r1") == "ollama"
    assert _guess_provider_from_model("claude-sonnet-4-6") == "anthropic"
    assert _guess_provider_from_model("gemini-2.5-flash") == "gemini"
    assert _guess_provider_from_model("unknown-model") is None


def test_route_decision_has_fallbacks():
    cfg = Config(mode="hybrid")
    router = Router(cfg)
    # Make only gemini available
    for name in router.providers:
        router.providers[name] = _StubProvider(name, available=(name == "gemini"))

    classification = classify("Explain bond convexity")
    decision = router.route(classification)
    # Since only gemini is available, no fallbacks
    assert decision.provider.name == "gemini"
