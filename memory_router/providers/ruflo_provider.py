"""Ruflo / multi-agent provider stub.

Optional. If a `ruflo` package is installed locally, this provider can hand
off complex agentic work to it. Until then, `is_available()` returns False
and the router will fall back to the next best option.
"""

from __future__ import annotations

from typing import List

from .base import BaseProvider, ProviderResult
from ..utils.tokens import estimate_messages_tokens, estimate_tokens


class RufloProvider(BaseProvider):
    name = "ruflo"

    def is_available(self) -> bool:
        try:
            import ruflo  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False

    def complete(self, model: str, messages: List[dict], **kwargs) -> ProviderResult:
        try:
            import ruflo  # type: ignore
        except ImportError as e:
            raise RuntimeError("ruflo not installed; install it to enable multi-agent mode") from e
        # Placeholder integration — adapt to your local Ruflo API.
        prompt = messages[-1]["content"] if messages else ""
        text = ruflo.run(prompt) if hasattr(ruflo, "run") else "[ruflo: no run() entrypoint found]"
        return ProviderResult(
            text=text,
            model=f"ruflo:{model}",
            input_tokens=estimate_messages_tokens(messages),
            output_tokens=estimate_tokens(text),
        )
