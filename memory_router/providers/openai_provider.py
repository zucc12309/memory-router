"""OpenAI provider stub.

Imports the official `openai` SDK lazily so the package works without it
installed. Install with: `pip install memory-router[openai]`.
"""

from __future__ import annotations

from typing import List

from .base import BaseProvider, ProviderResult
from ..security.keychain import get_secret
from ..utils.tokens import estimate_messages_tokens, estimate_tokens


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self):
        self._client = None
        self._api_key = get_secret("openai")

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. Run: pip install memory-router[openai]"
            ) from e
        if not self._api_key:
            raise RuntimeError(
                "No OpenAI API key found. Run `memory-router auth openai` to add one."
            )
        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def complete(self, model: str, messages: List[dict], **kwargs) -> ProviderResult:
        client = self._ensure_client()
        resp = client.chat.completions.create(model=model, messages=messages)
        text = resp.choices[0].message.content or ""
        # Use SDK usage if available; otherwise estimate.
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) or estimate_messages_tokens(messages)
        out_tok = getattr(usage, "completion_tokens", None) or estimate_tokens(text)
        return ProviderResult(text=text, model=model, input_tokens=in_tok, output_tokens=out_tok)
