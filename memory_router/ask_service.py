"""Core ask-question orchestration service.

Extracts the pipeline logic from cli.py so both CLI and MCP can share
the same classify → build-context → route → complete flow without
duplicating business logic.

This is NOT a UI layer — it returns structured results, never prints.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .adaptive_router import AdaptiveRouter, RouteOutcome
from .classifier import Classification, classify
from .config import Config
from .context_builder import build_context
from .memory.auto_capture import capture_turn
from .memory.sqlite_store import ConversationStore, MemoryStore, Message
from .router import Router
from .utils.logging import get_logger
from .utils.tokens import estimate_cost_usd, percent_saved

_log = get_logger(__name__)


@dataclass
class AskResult:
    """Structured result of the ask pipeline."""

    answer: str
    classification: Classification
    route_decision_reason: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    full_history_tokens: int
    sent_tokens: int
    latency_ms: int
    cost_usd: float
    token_savings_pct: float
    memories_used: int
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    fallback_used: bool = False
    error: Optional[str] = None


class AskService:
    """Stateless orchestrator for the ask pipeline.

    Usage::

        svc = AskService(cfg)
        result = svc.ask("How do I use pytest fixtures?")
    """

    def __init__(
        self,
        cfg: Config,
        mem_store: Optional[MemoryStore] = None,
        conv_store: Optional[ConversationStore] = None,
        router: Optional[Router] = None,
    ):
        self.cfg = cfg
        self._mem_store = mem_store
        self._conv_store = conv_store
        self._router = router

    @property
    def mem_store(self) -> MemoryStore:
        if self._mem_store is None:
            self._mem_store = MemoryStore()
        return self._mem_store

    @property
    def conv_store(self) -> ConversationStore:
        if self._conv_store is None:
            self._conv_store = ConversationStore()
        return self._conv_store

    @property
    def router(self) -> Router:
        if self._router is None:
            if self.cfg.adaptive_routing:
                self._router = AdaptiveRouter(self.cfg)
            else:
                self._router = Router(self.cfg)
        return self._router

    def ask(
        self,
        query: str,
        *,
        use_memory: bool = True,
        force_local: bool = False,
        session_id: str = "default",
        override_provider: Optional[str] = None,
        override_model: Optional[str] = None,
        mycelium=None,
    ) -> AskResult:
        """Run the full ask pipeline: classify → context → route → complete.

        Returns a structured AskResult. Never raises for provider errors —
        captures them in result.error instead.
        """
        request_id = uuid.uuid4().hex[:12]
        t0 = time.time()

        # 1. Classify
        classification = classify(query)

        # 2. Apply decay if enabled
        if self.cfg.memory_decay_enabled and use_memory:
            try:
                from .memory.decay import apply_decay
                apply_decay(self.mem_store)
            except Exception as e:
                _log.warning("decay failed", extra={"request_id": request_id, "error": str(e)})

        # 3. Build context
        built = build_context(
            query=query,
            classification=classification,
            cfg=self.cfg,
            mem_store=self.mem_store,
            conv_store=self.conv_store,
            use_memory=use_memory,
            session_id=session_id,
            mycelium=mycelium,
        )

        # 4. Route
        decision = self.router.route(
            classification,
            force_local=force_local,
            override_provider=override_provider,
            override_model=override_model,
        )

        # 5. Complete with fallback
        actual_provider = decision.provider.name
        actual_model = decision.model
        try:
            if hasattr(self.router, "complete_with_fallback"):
                result, actual_provider, actual_model = self.router.complete_with_fallback(
                    decision, built.messages
                )
            else:
                result = decision.provider.complete(decision.model, built.messages)
            answer = result.text or ""
            real_in = result.input_tokens or built.sent_tokens
            real_out = result.output_tokens or 0
            error = None
        except Exception as e:
            answer = ""
            real_in = built.sent_tokens
            real_out = 0
            error = str(e)
            _log.error("completion failed", extra={
                "request_id": request_id,
                "provider": actual_provider,
                "model": actual_model,
                "error": error,
            })

        latency_ms = int((time.time() - t0) * 1000)
        cost = estimate_cost_usd(actual_model, real_in, real_out)
        naive_in = built.full_history_tokens or built.sent_tokens
        savings = percent_saved(naive_in, real_in)

        # 6. Record adaptive outcome
        self._record_outcome(
            decision, classification, actual_provider, actual_model,
            real_in, real_out, latency_ms, cost, error, answer,
        )

        # 7. Auto-capture memory
        if use_memory and not error:
            try:
                capture_turn(
                    query=query,
                    answer=answer,
                    classification=classification,
                    cfg=self.cfg,
                    store=self.mem_store,
                    allow_capture=True,
                )
            except Exception as e:
                _log.warning("auto-capture failed", extra={
                    "request_id": request_id, "error": str(e),
                })

        # 8. Log conversation turn
        try:
            self.conv_store.add(Message(session_id=session_id, role="user", content=query))
            self.conv_store.add(Message(session_id=session_id, role="assistant", content=answer))
        except Exception as e:
            _log.warning("conversation log failed", extra={
                "request_id": request_id, "error": str(e),
            })

        return AskResult(
            answer=answer,
            classification=classification,
            route_decision_reason=decision.reason,
            provider=actual_provider,
            model=actual_model,
            input_tokens=real_in,
            output_tokens=real_out,
            full_history_tokens=naive_in,
            sent_tokens=built.sent_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            token_savings_pct=savings,
            memories_used=len(built.used_memories),
            request_id=request_id,
            fallback_used=(actual_provider != decision.provider.name),
            error=error,
        )

    def _record_outcome(
        self, decision, classification, actual_provider, actual_model,
        input_tokens, output_tokens, latency_ms, cost, error, answer,
    ):
        """Record adaptive routing outcome with improved quality signal."""
        if not self.cfg.adaptive_routing:
            return
        try:
            if isinstance(self.router, AdaptiveRouter):
                quality = _estimate_quality(answer, error, latency_ms)
                self.router.record_outcome(RouteOutcome(
                    provider=actual_provider,
                    model=actual_model,
                    task=classification.task,
                    domain=classification.domain,
                    complexity=classification.complexity,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                    quality_signal=quality,
                    error=error,
                ))
        except Exception as e:
            _log.debug("outcome recording failed", extra={"error": str(e)})


def _estimate_quality(answer: str, error: Optional[str], latency_ms: int) -> float:
    """Heuristic quality signal based on response characteristics.

    This replaces the synthetic 0.7/0.1 signal with a multi-factor estimate:
    - Error → 0.0
    - Empty/very short answer → 0.2
    - Reasonable length → 0.5–0.9 scaled by content
    - Penalty for extreme latency (>30s)
    """
    if error:
        return 0.0

    if not answer or len(answer.strip()) < 10:
        return 0.2

    # Base quality from answer length (diminishing returns)
    length = len(answer)
    if length < 50:
        base = 0.4
    elif length < 200:
        base = 0.6
    elif length < 1000:
        base = 0.75
    else:
        base = 0.85

    # Latency penalty for very slow responses
    if latency_ms > 30000:
        base *= 0.8
    elif latency_ms > 15000:
        base *= 0.9

    return min(1.0, base)
