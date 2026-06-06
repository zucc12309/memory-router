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

    result, used_provider, used_model = router.complete_with_fallback(decision, [{"role": "user", "content": "hi"}])
    assert result.text == "ok"
    assert used_provider == "gemini"
    assert used_model == "gemini-2.5-pro"


def test_no_fallback_raises():
    cfg = Config(mode="api")
    router = Router(cfg)

    primary = _StubProvider("anthropic", available=True, fail_on_complete=True)
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


def test_pick_fallback_model_respects_mini_tier():
    cfg = Config(mode="api")
    router = Router(cfg)
    assert router._pick_fallback_model("anthropic", "gpt-4o-mini") == cfg.models["anthropic_small"]


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


def test_default_provider_is_preferred_for_simple_queries():
    cfg = Config(mode="api", default_provider="openai")
    router = Router(cfg)

    for name in router.providers:
        router.providers[name] = _StubProvider(name, available=True)

    classification = classify("What is a parser?")
    decision = router.route(classification)

    assert decision.provider.name == "openai"


def test_local_model_preference_is_used_for_local_mode():
    cfg = Config(mode="local", local_model="qwen2.5:14b")
    router = Router(cfg)

    classification = classify("Explain APIs")
    decision = router.route(classification)

    assert decision.provider.name == "ollama"
    assert decision.model == "qwen2.5:14b"


def test_invalid_local_model_preference_falls_back_to_default():
    cfg = Config(mode="local", local_model="yes")
    router = Router(cfg)

    classification = classify("Explain APIs")
    decision = router.route(classification)

    assert decision.provider.name == "ollama"
    assert decision.model == cfg.models["local_simple"]


def test_pinned_routes_do_not_fallback():
    cfg = Config(mode="api")
    router = Router(cfg)

    primary = _StubProvider("anthropic", available=True, fail_on_complete=True)
    fallback = _StubProvider("gemini", available=True, fail_on_complete=False)
    router.providers["gemini"] = fallback

    from memory_router.router import RouteDecision

    decision = RouteDecision(
        provider=primary,
        model="claude-sonnet-4-6",
        reason="pinned test",
        fallback_providers=["gemini"],
        allow_fallback=False,
    )

    import pytest

    with pytest.raises(RuntimeError):
        router.complete_with_fallback(decision, [{"role": "user", "content": "hi"}])
