"""Integration tests for AskService.ask() with mocked router/providers.

Tests the full pipeline: classify → context → route → complete → result,
verifying orchestration, error handling, fallback marking, and outcome recording.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from memory_router.ask_service import AskService, AskResult
from memory_router.config import Config
from memory_router.memory.sqlite_store import MemoryStore, ConversationStore
from memory_router.providers.base import ProviderResult


def _make_service(tmp_path, adaptive=False):
    """Create an AskService with real stores and a mocked router."""
    cfg = Config()
    cfg.adaptive_routing = adaptive
    cfg.memory_enabled = True
    cfg.memory_decay_enabled = False
    cfg.auto_capture_memories = False

    mem = MemoryStore(path=tmp_path / "mem.sqlite")
    conv = ConversationStore(path=tmp_path / "conv.sqlite")

    mock_router = MagicMock()
    mock_provider = MagicMock()
    mock_provider.name = "mock"
    mock_decision = MagicMock()
    mock_decision.provider = mock_provider
    mock_decision.model = "mock-model"
    mock_decision.reason = "test routing"
    mock_router.route.return_value = mock_decision

    mock_result = ProviderResult(
        text="The answer is 42",
        model="mock-model",
        input_tokens=20,
        output_tokens=10,
    )
    mock_router.complete_with_fallback.return_value = (mock_result, "mock", "mock-model")

    svc = AskService(cfg, mem_store=mem, conv_store=conv, router=mock_router)
    return svc, mock_router


class TestAskPipeline:
    def test_returns_ask_result(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = svc.ask("What is the meaning of life?")
        assert isinstance(result, AskResult)
        assert result.answer == "The answer is 42"
        assert result.provider == "mock"
        assert result.model == "mock-model"
        assert result.error is None

    def test_request_id_unique(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        r1 = svc.ask("q1")
        r2 = svc.ask("q2")
        assert r1.request_id != r2.request_id

    def test_classification_populated(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = svc.ask("Write a Python function to sort a list")
        assert result.classification is not None
        assert result.classification.task in ("code", "explain", "reason", "chat", "translate", "summarize")

    def test_token_savings_calculated(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = svc.ask("hello")
        assert result.sent_tokens >= 0
        assert result.full_history_tokens >= 0

    def test_latency_tracked(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = svc.ask("test query")
        assert result.latency_ms >= 0

    def test_conversation_logged(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        svc.ask("hello there", session_id="test-sess")
        turns = svc.conv_store.recent("test-sess", limit=10)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[0].content == "hello there"
        assert turns[1].role == "assistant"

    def test_error_captured_not_raised(self, tmp_path):
        svc, mock_router = _make_service(tmp_path)
        mock_router.complete_with_fallback.side_effect = ConnectionError("network down")

        result = svc.ask("broken query")
        assert result.error is not None
        assert "network down" in result.error
        assert result.answer == ""

    def test_force_local(self, tmp_path):
        svc, mock_router = _make_service(tmp_path)
        svc.ask("test", force_local=True)
        call_kwargs = mock_router.route.call_args
        assert call_kwargs.kwargs.get("force_local") is True

    def test_override_provider(self, tmp_path):
        svc, mock_router = _make_service(tmp_path)
        svc.ask("test", override_provider="anthropic")
        call_kwargs = mock_router.route.call_args
        assert call_kwargs.kwargs.get("override_provider") == "anthropic"

    def test_override_model(self, tmp_path):
        svc, mock_router = _make_service(tmp_path)
        svc.ask("test", override_model="gpt-4o")
        call_kwargs = mock_router.route.call_args
        assert call_kwargs.kwargs.get("override_model") == "gpt-4o"

    def test_fallback_detection(self, tmp_path):
        svc, mock_router = _make_service(tmp_path)
        # Simulate fallback: returned provider differs from decision
        mock_result = ProviderResult(text="fallback answer", model="ollama-llama3")
        mock_router.complete_with_fallback.return_value = (mock_result, "ollama", "llama3")

        result = svc.ask("test")
        assert result.fallback_used is True
        assert result.provider == "ollama"

    def test_memory_disabled(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = svc.ask("test", use_memory=False)
        assert result.answer == "The answer is 42"
        assert result.memories_used == 0

    def test_decay_applied_when_enabled(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        svc.cfg.memory_decay_enabled = True
        with patch("memory_router.memory.decay.apply_decay") as mock_decay:
            mock_decay.return_value = 0
            svc.ask("test")
            mock_decay.assert_called_once()

    def test_decay_failure_non_fatal(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        svc.cfg.memory_decay_enabled = True
        with patch("memory_router.memory.decay.apply_decay", side_effect=RuntimeError("db locked")):
            result = svc.ask("test")
            assert result.error is None  # Decay failure doesn't set error

    def test_auto_capture_on_success(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        svc.cfg.auto_capture_memories = True
        with patch("memory_router.ask_service.capture_turn") as mock_capture:
            svc.ask("test")
            mock_capture.assert_called_once()

    def test_auto_capture_skipped_on_error(self, tmp_path):
        svc, mock_router = _make_service(tmp_path)
        svc.cfg.auto_capture_memories = True
        mock_router.complete_with_fallback.side_effect = RuntimeError("fail")
        with patch("memory_router.ask_service.capture_turn") as mock_capture:
            svc.ask("test")
            mock_capture.assert_not_called()


class TestAskServiceRouterCreation:
    def test_creates_adaptive_router_when_enabled(self, tmp_path):
        cfg = Config()
        cfg.adaptive_routing = True
        mem = MemoryStore(path=tmp_path / "m.sqlite")
        conv = ConversationStore(path=tmp_path / "c.sqlite")
        svc = AskService(cfg, mem_store=mem, conv_store=conv)
        from memory_router.adaptive_router import AdaptiveRouter
        assert isinstance(svc.router, AdaptiveRouter)

    def test_creates_basic_router_when_disabled(self, tmp_path):
        cfg = Config()
        cfg.adaptive_routing = False
        mem = MemoryStore(path=tmp_path / "m.sqlite")
        conv = ConversationStore(path=tmp_path / "c.sqlite")
        svc = AskService(cfg, mem_store=mem, conv_store=conv)
        from memory_router.router import Router
        assert isinstance(svc.router, Router)
