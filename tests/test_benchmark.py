from __future__ import annotations

from types import SimpleNamespace

from memory_router import benchmark
from memory_router.config import Config
from memory_router.providers.base import ProviderResult
from memory_router.utils.tokens import estimate_messages_tokens, estimate_tokens


def _make_case() -> benchmark.BenchmarkCase:
    history = []
    for idx in range(12):
        history.append(
            {
                "role": "user",
                "content": (
                    f"Keep the implementation small, readable, and easy to review. "
                    f"Filler turn {idx}."
                ),
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": (
                    "Agreed, we should avoid extra abstraction and keep the code paths "
                    "straightforward."
                ),
            }
        )

    return benchmark.BenchmarkCase(
        name="pytest case",
        query="Write tests for the parser helper in this repo.",
        history=history,
        memories=[
            {
                "task": "code",
                "domain": "software",
                "concepts": ["pytest", "tests"],
                "content": "Prefer pytest for tests and keep them isolated.",
                "importance": 0.95,
            }
        ],
        must_include=["pytest"],
        must_avoid=["unittest"],
    )


def test_benchmark_prompt_only_reports_token_savings():
    case = _make_case()
    record = benchmark.evaluate_case(case, cfg=Config(token_budget=120), run_model=False)

    assert record.status == "prompt-only"
    assert record.raw_tokens > record.optimized_tokens
    assert record.raw_saved_pct >= 0
    assert record.baseline_saved_pct >= 0


def test_benchmark_run_scores_quality_with_fake_provider(monkeypatch):
    case = _make_case()

    class FakeProvider:
        name = "fake"

        def is_available(self) -> bool:
            return True

        def complete(self, model: str, messages, **kwargs):
            text = "Use pytest and keep the tests isolated." if any(
                "pytest" in (m.get("content", "") or "").lower() for m in messages
            ) else "Use unittest."
            return ProviderResult(
                text=text,
                model=model,
                input_tokens=estimate_messages_tokens(messages),
                output_tokens=estimate_tokens(text),
            )

    class FakeRouter:
        def __init__(self, cfg):
            self.cfg = cfg

        def route(self, classification, force_local=False):
            return SimpleNamespace(provider=FakeProvider(), model="fake-model", reason="fake route")

    monkeypatch.setattr(benchmark, "Router", FakeRouter)

    record = benchmark.evaluate_case(case, cfg=Config(token_budget=400), run_model=True)

    assert record.status == "ok"
    assert record.provider == "fake"
    assert record.model == "fake-model"
    assert record.baseline_score == 0.0
    assert record.optimized_score == 1.0
    assert record.raw_tokens > record.optimized_tokens


def test_benchmark_local_mode_auto_starts_ollama(monkeypatch):
    case = _make_case()
    provider = None
    started = []
    model_checks = []

    class FakeProvider:
        name = "ollama"

        def __init__(self):
            self.available = False

        def is_available(self) -> bool:
            return self.available

        def complete(self, model: str, messages, **kwargs):
            return ProviderResult(
                text="Use pytest and keep the tests isolated.",
                model=model,
                input_tokens=estimate_messages_tokens(messages),
                output_tokens=estimate_tokens("Use pytest and keep the tests isolated."),
            )

    class FakeRouter:
        def __init__(self, cfg):
            self.cfg = cfg

        def route(self, classification, force_local=False):
            nonlocal provider
            provider = FakeProvider()
            return SimpleNamespace(provider=provider, model="llama3.1:8b", reason="local route")

    def fake_ensure_ollama_running(host: str, timeout: int = 15):
        started.append(host)
        assert provider is not None
        provider.available = True
        return True

    monkeypatch.setattr(benchmark, "Router", FakeRouter)
    monkeypatch.setattr(benchmark, "ensure_ollama_running", fake_ensure_ollama_running)
    monkeypatch.setattr(
        benchmark,
        "ensure_ollama_model_available",
        lambda host, model: model_checks.append((host, model)) or False,
    )

    record = benchmark.evaluate_case(case, cfg=Config(mode="local", token_budget=400), run_model=True, force_local=True)

    assert started == ["http://localhost:11434"]
    assert model_checks == [("http://localhost:11434", "llama3.1:8b")]
    assert record.status == "ok"
    assert record.provider == "ollama"
