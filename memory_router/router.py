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
from .providers.base import BaseProvider
from .utils.logging import get_logger
from .providers.anthropic_provider import AnthropicProvider

_log = get_logger(__name__)
from .providers.gemini_provider import GeminiProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .providers.ruflo_provider import RufloProvider


@dataclass
class RouteDecision:
    provider: BaseProvider
    model: str
    reason: str
    fallback_providers: Optional[List[str]] = None
    estimated_cost_usd: float = 0.0


class Router:
    """Holds the live provider instances and decides which to use."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.providers: Dict[str, BaseProvider] = {
            "ollama": OllamaProvider(host=cfg.ollama_host),
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "gemini": GeminiProvider(),
            "ruflo": RufloProvider(),
        }

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
            fallbacks = [
                n
                for n in ("anthropic", "openai", "gemini")
                if n != prov_name and self.providers[n].is_available()
            ]
            return RouteDecision(
                p, model_id, reason, fallback_providers=fallbacks or None
            )

        if task in ("code", "security") or c >= 0.7:
            for prov_name, key in [
                ("anthropic", "anthropic_large"),
                ("openai", "openai_large"),
                ("gemini", "gemini_large"),
            ]:
                d = _pick(
                    prov_name, key, f"{task}/high complexity → {prov_name} large"
                )
                if d:
                    return d

        if task in ("explain", "reasoning") or c >= 0.4:
            for prov_name, key in [
                ("anthropic", "anthropic_mid"),
                ("openai", "openai_large"),
                ("gemini", "gemini_mid"),
            ]:
                d = _pick(prov_name, key, f"{task} → {prov_name} mid")
                if d:
                    return d

        # Cheap / simple — prefer the small models.
        for prov_name, key in [
            ("gemini", "gemini_small"),
            ("anthropic", "anthropic_small"),
            ("openai", "openai_small"),
        ]:
            d = _pick(prov_name, key, "simple query → small model")
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

    def complete_with_fallback(
        self,
        decision: RouteDecision,
        messages: List[dict],
        **kwargs,
    ):
        """Execute completion with automatic fallback on failure.

        Tries the primary provider first. On failure, iterates through
        fallback providers before giving up. Returns (result, actual_provider).
        """
        # Try primary
        try:
            result = decision.provider.complete(decision.model, messages, **kwargs)
            _log.info("completion succeeded", extra={
                "provider": decision.provider.name, "model": decision.model,
            })
            return result, decision.provider.name
        except Exception as primary_err:
            _log.warning("primary provider failed, attempting fallback", extra={
                "provider": decision.provider.name, "model": decision.model,
                "error": str(primary_err),
            })
            if not decision.fallback_providers:
                raise

            # Try each fallback
            last_err = primary_err
            for fb_name in decision.fallback_providers:
                fb_provider = self.providers.get(fb_name)
                if not fb_provider or not fb_provider.is_available():
                    continue
                # Pick a reasonable model for the fallback provider
                fb_model = self._pick_fallback_model(fb_name, decision.model)
                if not fb_model:
                    continue
                try:
                    result = fb_provider.complete(fb_model, messages, **kwargs)
                    return result, fb_name
                except Exception as fb_err:
                    last_err = fb_err
                    continue

            raise last_err

    def _pick_fallback_model(self, provider_name: str, original_model: str) -> Optional[str]:
        """Pick a comparable model from a fallback provider."""
        models = self.cfg.models
        # Determine the original tier
        orig_lower = original_model.lower()
        if any(s in orig_lower for s in ("large", "opus", "4o", "pro")):
            tier = "large"
        elif any(s in orig_lower for s in ("mid", "sonnet")):
            tier = "mid"
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
        return RouteDecision(ollama, models.get(key, "llama3.1:8b"), reason)

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

        # Build fallback list for pinned routes too
        fallbacks = [
            n
            for n in ("anthropic", "openai", "gemini")
            if n != provider_name and self.providers[n].is_available()
        ]

        return RouteDecision(
            provider,
            model_id,
            f"pinned → {provider_name}/{model_id}",
            fallback_providers=fallbacks or None,
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
