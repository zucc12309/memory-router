"""Google Gemini provider stub.

Install with: `pip install memory-router[gemini]`. Uses the official
`google-generativeai` SDK. Gemini's API takes a system prompt out-of-band
(via `system_instruction`), so we split it from the messages list and convert
the OpenAI-style {role, content} format into Gemini's {role, parts} format.
"""

from __future__ import annotations

from typing import List

from .base import BaseProvider, ProviderResult
from ..security.keychain import get_secret
from ..utils.tokens import estimate_messages_tokens, estimate_tokens


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self):
        self._configured = False
        self._api_key = get_secret("gemini")

    def _ensure_configured(self):
        if self._configured:
            return
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "google-generativeai package not installed. "
                "Run: pip install memory-router[gemini]"
            ) from e
        if not self._api_key:
            raise RuntimeError(
                "No Gemini API key found. Run `memory-router auth gemini` to add one."
            )
        genai.configure(api_key=self._api_key)
        self._configured = True
        return genai

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import google.generativeai  # noqa: F401
            return True
        except ImportError:
            return False

    def complete(self, model: str, messages: List[dict], **kwargs) -> ProviderResult:
        import google.generativeai as genai  # type: ignore
        self._ensure_configured()

        # Pull system prompt out — Gemini takes it as `system_instruction`.
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat = [m for m in messages if m.get("role") != "system"]

        # Convert OpenAI roles to Gemini roles: assistant -> model, user -> user.
        gemini_history = []
        for m in chat[:-1]:  # everything except the latest user turn becomes history
            role = "model" if m["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [m["content"]]})

        latest = chat[-1]["content"] if chat else ""

        model_obj = genai.GenerativeModel(
            model_name=model,
            system_instruction="\n\n".join(system_parts) if system_parts else None,
        )
        chat_session = model_obj.start_chat(history=gemini_history)
        resp = chat_session.send_message(latest)

        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", None) or estimate_messages_tokens(messages)
        out_tok = getattr(usage, "candidates_token_count", None) or estimate_tokens(text)
        return ProviderResult(text=text, model=model, input_tokens=in_tok, output_tokens=out_tok)
