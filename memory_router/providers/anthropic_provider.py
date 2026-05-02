"""Anthropic provider stub.

Install with: `pip install memory-router[anthropic]`. Anthropic's SDK takes
the system prompt out-of-band, so we split it from the messages list before
sending.
"""

from __future__ import annotations

from typing import List

from .base import BaseProvider, ProviderResult
from ..security.keychain import get_secret
from ..utils.tokens import estimate_messages_tokens, estimate_tokens


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self):
        self._client = None
        self._api_key = get_secret("anthropic")

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install memory-router[anthropic]"
            ) from e
        if not self._api_key:
            raise RuntimeError(
                "No Anthropic API key found. Run `memory-router auth anthropic` to add one."
            )
        self._client = Anthropic(api_key=self._api_key)
        return self._client

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    def complete(self, model: str, messages: List[dict], **kwargs) -> ProviderResult:
        client = self._ensure_client()
        # Pull a system message out — Anthropic uses a top-level `system` field.
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat = [m for m in messages if m.get("role") != "system"]
        resp = client.messages.create(
            model=model,
            max_tokens=kwargs.get("max_tokens", 1024),
            system="\n\n".join(system_parts) if system_parts else "",
            messages=chat,
        )
        # The SDK returns a list of content blocks; concatenate text blocks.
        text = "".join(getattr(b, "text", "") for b in resp.content)
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "input_tokens", None) or estimate_messages_tokens(messages)
        out_tok = getattr(usage, "output_tokens", None) or estimate_tokens(text)
        return ProviderResult(text=text, model=model, input_tokens=in_tok, output_tokens=out_tok)
