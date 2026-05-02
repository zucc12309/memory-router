"""Provider interface.

Every backend (OpenAI, Anthropic, Ollama, Ruflo, ...) implements this small
contract. Keeping the surface tiny makes it trivial to add new providers
without touching the router.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class ProviderResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class BaseProvider(ABC):
    """Minimal LLM provider contract."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider can serve a request right now."""

    @abstractmethod
    def complete(self, model: str, messages: List[dict], **kwargs) -> ProviderResult:
        """Send `messages` (OpenAI-style {role, content}) and return a result."""
