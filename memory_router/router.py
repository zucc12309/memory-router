"""Model router.

Picks (provider, model) for a given classification + config. The rules are
deliberately readable — tweak them in one place to change routing behavior.

v2 changes:
  - Fallback routing: if the primary provider fails, try alternatives
  - Cost-aware model selection within tiers
  - Outcome recording for future adaptive routing
  - Retry with exponential backoff (single retry)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .classifier import Classification
from .config import Config
from .providers.anthropic_provider import AnthropicProvider
from .providers.base import BaseProvider
from .providers.gemini_provider import GeminiProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .providers.ruflo_provider import RufloProvider
from .utils.logging import get_logger
from .utils.system import normalize_ollama_model_name

_log = get_logger(__name__)

# Retry configuration
MAX_RETRIES = 2
RETRY_BASE_DELAY_S = 1.0
RETRY_BACKOFF_FACTOR = 2.0


@dataclass
class RouteDecision:
    provider: BaseProvider
    model: str
    reason: str
    fallback_providers: Optional[List[str]] = None
    estimated_cost_usd: float = 0.0
    allow_fallback: bool = True


# ---------------------------------------------------------------------------
# Provider registry — lazy instantiation, extensible
# ---------------------------------------------------------------------------

_PROVIDER_FACTORIES: Dict[str, type] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "ruflo": RufloProvider,
}


def register_provider(name: str, cls: type) -> None:
    """Register a custom provider class.

    Usage::

        from memory_router.router import register_provider
        register_provider("my_provider", MyProvider)
    """
    _PROVIDER_FACTORIES[name] = cls


def _build_providers(cfg: Config) -> Dict[str, BaseProvider]:
    """Instantiate all registered providers."""
    providers: Dict[str, BaseProvider] = {}
    for name, factory in _PROVIDER_FACTORIES.items():
        try:
            if name == "ollama":
                providers[name] = factory(host=cfg.ollama_host)
            else:
                providers[name] = factory()
        except Exception:
            pass  # Skip providers that fail to instantiate
    return providers


class Router:
    """Holds the live provider instances and decides which to use."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.providers: Dict[str, BaseProvider] = _build_providers(cfg)

    def _ordered_provider_names(self, names: List[str]) -> List[str]:
        """Prefer the configured default provider when it is in the candidate set."""
        preferred = self.cfg.default_provider
        ordered = [n for n in names if n != preferred]
        if preferred in names:
            return [preferred, *ordered]
        return ordered

    def route(
        self,
        classification: Classification,
        force_local: bool = False,
        override_provider: Optional[str] = None,
        override_model: Optional[str] = None,
    ) -> RouteDecision:
        """Pick a provider+model based on overrides, mode, classification, availability."""
        mode = self.cfg.mode
        models = self.cfg.models

        # 1. Per-call overrides win over everything.
        # 2. Then config-level pins (force_provider / force_model).
        # 3. Then auto-routing.
        pinned_provider = override_provider or self.cfg.force_provider or None
        pinned_model = override_model or self.cfg.force_model or None

        if pinned_provider or pinned_model:
            return self._route_pinned(pinned_provider, pinned_model, classification)

        # --no-memory + privacy-leaning users may also want forced-local routing.
        if force_local or mode == "local":
            return self._route_local(classification, "local-only mode")

        # Ruflo mode: prefer agentic provider for multi-step / agentic tasks.
        if mode == "ruflo" and classification.task == "agentic":
            ruflo = self.providers["ruflo"]
            if ruflo.is_available():
                return RouteDecision(
                    ruflo,
                    models.get("local_default", "ruflo"),
                    "agentic task → ruflo",
                )

        # Hybrid + API modes share the rule table; difference is hybrid prefers
        # local for simple queries.
        if mode == "hybrid" and classification.complexity < 0.3:
            local = self._route_local(classification, "low complexity → local")
            if local.provider.is_available():
                return local

        # Map (task, complexity) → (provider, model size).
        task = classification.task
        c = classification.complexity

        # Sensible per-provider fallbacks so a stale config can never KeyError.
        _DEFAULT_FALLBACKS = {
            "anthropic_small": "claude-haiku-4-5-20251001",
            "anthropic_mid": "claude-sonnet-4-6",
            "anthropic_large": "claude-opus-4-7",
            "openai_small": "gpt-4o-mini",
            "openai_large": "gpt-4o",
            "gemini_small": "gemini-2.5-flash",
            "gemini_mid": "gemini-2.5-pro",
            "gemini_large": "gemini-2.5-pro",
        }

        def _pick(prov_name: str, key: str, reason: str):
            p = self.providers[prov_name]
            if not p.is_available():
                return None
            model_id = models.get(key) or _DEFAULT_FALLBACKS.get(key)
            if not model_id:
                return None
            # Build fallback list (other available providers at same tier)
            fallbacks = self._ordered_provider_names([
                n for n in ("anthropic", "openai", "gemini") if n != prov_name
            ])
            fallbacks = [n for n in fallbacks if self.providers[n].is_available()]
            return RouteDecision(
                p,
                model_id,
                reason,
                fallback_providers=fallbacks or None,
            )

        if task in ("code", "security") or c >= 0.7:
            for prov_name in self._ordered_provider_names([
                "anthropic", "openai", "gemini"
            ]):
                key = f"{prov_name}_large"
                d = _pick(prov_name, key, f"{task}/high complexity → {prov_name} large")
                if d:
                    return d

        if task in ("explain", "reasoning") or c >= 0.4:
            for prov_name in self._ordered_provider_names([
                "anthropic", "openai", "gemini"
            ]):
                key = "openai_large" if prov_name == "openai" else f"{prov_name}_mid"
                d = _pick(prov_name, key, f"{task} → {prov_name} mid")
                if d:
                    return d

        # Cheap / simple — prefer the small models.
        for prov_name in self._ordered_provider_names([
            "gemini", "anthropic", "openai"
        ]):
            model_key = f"{prov_name}_small"
            d = _pick(prov_name, model_key, "simple query → small model")
            if d:
                return d

        # No API providers available — fall back to local or fail hard in api mode.
        if mode == "api":
            missing = []
            for name in ("openai", "anthropic", "gemini"):
                p = self.providers[name]
                if not p.is_available():
                    missing.append(name)
            raise RuntimeError(
                "API mode is set but no API provider is ready. "
                f"Unavailable: {', '.join(missing) or 'all'}. "
                "Most likely the SDK isn't installed or the API key isn't saved.\n"
                "  • SDKs:  pip install \"memory-router[all]\"\n"
                "  • Keys:  memory-router auth openai | anthropic | gemini\n"
                "  • Check: memory-router doctor"
            )
        return self._route_local(
            classification, "no API providers available, falling back to local"
        )

    def _attempt_with_retry(
        self, provider: BaseProvider, model: str, messages: List[dict], **kwargs
    ):
        """Try a single provider with exponential backoff retry."""
        last_err = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                t0 = time.time()
                result = provider.complete(model, messages, **kwargs)
                result.latency_ms = int((time.time() - t0) * 1000)
                return result
            except Exception as e:
                last_err = e
                is_retryable = _is_retryable_error(e)
                if not is_retryable or attempt >= MAX_RETRIES:
                    raise
                delay = RETRY_BASE_DELAY_S * (RETRY_BACKOFF_FACTOR ** attempt)
                _log.warning(
                    "retrying provider after error",
                    extra={
                        "provider": provider.name, "model": model,
                        "attempt": attempt + 1, "delay_s": delay,
                        "error": str(e),
                    },
                )
                time.sleep(delay)
        raise last_err  # unreachable but satisfies type checker

    def complete_with_fallback(
        self,
        decision: RouteDecision,
        messages: List[dict],
        **kwargs,
    ):
        """Execute completion with retry + automatic fallback on failure.

        Tries the primary provider first (with retries for transient errors).
        On permanent failure, iterates through fallback providers before
        giving up. Returns (result, actual_provider, actual_model).
        """
        # Try primary with retry
        try:
            result = self._attempt_with_retry(
                decision.provider, decision.model, messages, **kwargs
            )
            _log.info("completion succeeded", extra={
                "provider": decision.provider.name, "model": decision.model,
            })
            return result, decision.provider.name, decision.model
        except Exception as primary_err:
            if not getattr(decision, "allow_fallback", True):
                raise
            _log.warning("primary provider failed, attempting fallback", extra={
                "provider": decision.provider.name, "model": decision.model,
                "error": str(primary_err),
            })
            if not decision.fallback_providers:
                raise

            # Try each fallback (also with retry)
            last_err = primary_err
            for fb_name in decision.fallback_providers:
                fb_provider = self.providers.get(fb_name)
                if not fb_provider or not fb_provider.is_available():
                    continue
                fb_model = self._pick_fallback_model(fb_name, decision.model)
                if not fb_model:
                    continue
                try:
                    result = self._attempt_with_retry(
                        fb_provider, fb_model, messages, **kwargs
                    )
                    _log.info("fallback succeeded", extra={
                        "provider": fb_name, "model": fb_model,
                    })
                    return result, fb_name, fb_model
                except Exception as fb_err:
                    last_err = fb_err
                    continue

            raise last_err

    def _pick_fallback_model(self, provider_name: str, original_model: str) -> Optional[str]:
        """Pick a comparable model from a fallback provider."""
        models = self.cfg.models
        # Determine the original tier
        orig_lower = original_model.lower()
        if any(s in orig_lower for s in ("mini", "flash", "haiku", "small", "nano", "tiny")):
            tier = "small"
        elif any(s in orig_lower for s in ("large", "opus")):
            tier = "large"
        elif any(s in orig_lower for s in ("mid", "sonnet", "pro")):
            tier = "mid"
        elif any(s in orig_lower for s in ("4o", "o3", "o4")):
            tier = "large"
        else:
            tier = "small"

        key = f"{provider_name}_{tier}"
        return models.get(key) or models.get(f"{provider_name}_mid") or models.get(f"{provider_name}_small")

    def _route_local(
        self, classification: Classification, reason: str
    ) -> RouteDecision:
        models = self.cfg.models
        ollama = self.providers["ollama"]
        key = "local_simple" if classification.complexity < 0.3 else "local_default"
        configured_model = normalize_ollama_model_name(self.cfg.local_model)
        model_id = configured_model or models.get(key, "llama3.1:8b")
        return RouteDecision(ollama, model_id, reason)

    def _route_pinned(
        self,
        provider_name: Optional[str],
        model_id: Optional[str],
        classification: Classification,
    ) -> RouteDecision:
        """Honor a pinned provider/model. Infers the missing side when needed."""
        if not provider_name and model_id:
            provider_name = _guess_provider_from_model(model_id)
            if not provider_name:
                raise RuntimeError(
                    f"Couldn't infer a provider for model '{model_id}'. "
                    "Pass --provider or set force_provider."
                )

        if provider_name not in self.providers:
            raise RuntimeError(
                f"Unknown provider '{provider_name}'. "
                f"Valid options: {', '.join(self.providers)}."
            )

        provider = self.providers[provider_name]
        if not provider.is_available():
            raise RuntimeError(
                f"Provider '{provider_name}' is not available — "
                "missing SDK or API key. Run `memory-router doctor`."
            )

        if not model_id:
            tier = (
                "small"
                if classification.complexity < 0.3
                else ("large" if classification.complexity >= 0.7 else "mid")
            )
            tier_key = f"{provider_name}_{tier}"
            model_id = (
                self.cfg.models.get(tier_key)
                or self.cfg.models.get(f"{provider_name}_mid")
                or self.cfg.models.get(f"{provider_name}_small")
            )
        if not model_id:
            raise RuntimeError(
                f"No model id resolved for provider '{provider_name}'. "
                "Pass --model or set force_model."
            )

        return RouteDecision(
            provider,
            model_id,
            f"pinned → {provider_name}/{model_id}",
            fallback_providers=None,
            allow_fallback=False,
        )


def _guess_provider_from_model(model_id: str) -> Optional[str]:
    """Cheap heuristic: infer the provider from a model id substring."""
    m = (model_id or "").lower()
    if m.startswith("gpt") or "openai" in m or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "gemini"
    if "llama" in m or "mistral" in m or "qwen" in m or "phi" in m or "deepseek" in m:
        return "ollama"
    return None


def _is_retryable_error(exc: Exception) -> bool:
    """Determine if an error is transient and worth retrying.

    Rate-limit (429), server errors (5xx), timeouts, and connection
    errors are retryable.  Auth (401/403) and bad-request (400) are not.
    """
    msg = str(exc).lower()
    # HTTP status codes embedded in error messages
    if "429" in msg or "rate" in msg:
        return True
    if any(code in msg for code in ("500", "502", "503", "504")):
        return True
    # Connection / timeout errors
    if any(term in msg for term in ("timeout", "timed out", "connection", "reset")):
        return True
    # Provider-specific retryable attributes
    if hasattr(exc, "status_code"):
        code = getattr(exc, "status_code", 0)
        return code in (429, 500, 502, 503, 504)
    return False
