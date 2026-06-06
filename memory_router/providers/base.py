"""Provider interface.

Every backend (OpenAI, Anthropic, Ollama, Ruflo, ...) implements this small
contract. Keeping the surface tiny makes it trivial to add new providers
without touching the router.

v2: Added streaming support via `stream()` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, List, Optional


@dataclass
class ProviderResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    text: str
    finished: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class BaseProvider(ABC):
    """Minimal LLM provider contract."""

    name: str = "base"

    @staticmethod
    def split_system_messages(messages: List[dict]):
        """Split system messages from chat messages.

        Returns (system_text, chat_messages) where system_text is the
        concatenated system prompts and chat_messages excludes them.
        """
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat = [m for m in messages if m.get("role") != "system"]
        system_text = "\n\n".join(system_parts) if system_parts else ""
        return system_text, chat

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider can serve a request right now."""

    @abstractmethod
    def complete(self, model: str, messages: List[dict], **kwargs) -> ProviderResult:
        """Send `messages` (OpenAI-style {role, content}) and return a result."""

    def stream(
        self, model: str, messages: List[dict], **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """Stream response chunks. Default falls back to non-streaming complete().

        Override in subclasses to provide real streaming support.
        """
        result = self.complete(model, messages, **kwargs)
        yield StreamChunk(
            text=result.text,
            finished=True,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
