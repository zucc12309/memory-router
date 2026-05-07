"""Google Gemini provider — uses the modern `google-genai` SDK.

The legacy `google-generativeai` package is deprecated, so we use
`google-genai` (Google GenAI SDK). Install with:
    pip install memory-router[gemini]

Gemini takes the system prompt out-of-band via GenerateContentConfig, and uses
{role: "user"|"model", parts: [...]} instead of OpenAI's {role, content}.
"""

from __future__ import annotations

from typing import List

from .base import BaseProvider, ProviderResult
from ..security.keychain import get_secret
from ..utils.tokens import estimate_messages_tokens, estimate_tokens


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self):
        self._client = None
        self._api_key = get_secret("gemini")

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "google-genai package not installed. Run: pip install memory-router[gemini]"
            ) from e
        if not self._api_key:
            raise RuntimeError(
                "No Gemini API key found. Run `memory-router auth gemini` to add one."
            )
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            from google import genai  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False

    def complete(self, model: str, messages: List[dict], **kwargs) -> ProviderResult:
        client = self._ensure_client()
        from google.genai import types  # type: ignore

        # Pull system prompt(s) — Gemini takes them as `system_instruction`.
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat = [m for m in messages if m.get("role") != "system"]

        # Convert history into Gemini's content/parts shape.
        # OpenAI roles: user / assistant / system  →  Gemini roles: user / model.
        history = []
        for m in chat[:-1]:
            role = "model" if m["role"] == "assistant" else "user"
            history.append({"role": role, "parts": [{"text": m["content"]}]})

        latest = chat[-1]["content"] if chat else ""

        config_kwargs = {}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)
        cfg = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        chat_session = client.chats.create(model=model, history=history, config=cfg)
        resp = chat_session.send_message(latest)

        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", None) or estimate_messages_tokens(messages)
        out_tok = getattr(usage, "candidates_token_count", None) or estimate_tokens(text)
        return ProviderResult(text=text, model=model, input_tokens=in_tok, output_tokens=out_tok)
