"""Model router.

Picks (provider, model) for a given classification + config. The rules are
deliberately readable — tweak them in one place to change routing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .classifier import Classification
from .config import Config
from .providers.base import BaseProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.gemini_provider import GeminiProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .providers.ruflo_provider import RufloProvider


@dataclass
class RouteDecision:
    provider: BaseProvider
    model: str
    reason: str


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

    def route(self, classification: Classification, force_local: bool = False) -> RouteDecision:
        """Pick a provider+model based on mode, classification, and availability."""
        mode = self.cfg.mode
        models = self.cfg.models

        # --no-memory + privacy-leaning users may also want forced-local routing.
        if force_local or mode == "local":
            return self._route_local(classification, "local-only mode")

        # Ruflo mode: prefer agentic provider for multi-step / agentic tasks.
        if mode == "ruflo" and classification.task == "agentic":
            ruflo = self.providers["ruflo"]
            if ruflo.is_available():
                return RouteDecision(ruflo, models.get("local_default", "ruflo"), "agentic task → ruflo")

        # Hybrid + API modes share the rule table; difference is hybrid prefers
        # local for simple queries.
        if mode == "hybrid" and classification.complexity < 0.3:
            local = self._route_local(classification, "low complexity → local")
            if local.provider.is_available():
                return local

        # Map (task, complexity) → (provider, model size).
        task = classification.task
        c = classification.complexity

        if task in ("code", "security") or c >= 0.7:
            for prov_name, key in [
                ("anthropic", "anthropic_large"),
                ("openai", "openai_large"),
                ("gemini", "gemini_large"),
            ]:
                p = self.providers[prov_name]
                if p.is_available():
                    return RouteDecision(p, models[key], f"{task}/high complexity → {prov_name} large")

        if task in ("explain", "reasoning") or c >= 0.4:
            for prov_name, key in [
                ("anthropic", "anthropic_mid"),
                ("openai", "openai_large"),
                ("gemini", "gemini_mid"),
            ]:
                p = self.providers[prov_name]
                if p.is_available():
                    return RouteDecision(p, models[key], f"{task} → {prov_name} mid")

        # Cheap / simple — prefer the small models.
        for prov_name, key in [
            ("gemini", "gemini_small"),
            ("anthropic", "anthropic_small"),
            ("openai", "openai_small"),
        ]:
            p = self.providers[prov_name]
            if p.is_available():
                return RouteDecision(p, models[key], "simple query → small model")

        # No API providers available — fall back to local.
        return self._route_local(classification, "no API providers available, falling back to local")

    def _route_local(self, classification: Classification, reason: str) -> RouteDecision:
        models = self.cfg.models
        ollama = self.providers["ollama"]
        key = "local_simple" if classification.complexity < 0.3 else "local_default"
        return RouteDecision(ollama, models.get(key, "llama3.1:8b"), reason)
