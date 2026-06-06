"""Google Gemini provider with streaming support.

Uses the modern `google-genai` SDK. Install with:
    pip install memory-router[gemini]

Gemini takes the system prompt out-of-band via GenerateContentConfig, and uses
{role: "user"|"model", parts: [...]} instead of OpenAI's {role, content}.
"""

from __future__ import annotations

from typing import Generator, List

from .base import BaseProvider, ProviderResult, StreamChunk
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

    def _prepare_messages(self, messages: List[dict]):
        """Split system/chat and convert to Gemini format."""
        system_text, chat = self.split_system_messages(messages)
        system_parts = [system_text] if system_text else []

        history = []
        for m in chat[:-1]:
            role = "model" if m["role"] == "assistant" else "user"
            history.append({"role": role, "parts": [{"text": m["content"]}]})

        latest = chat[-1]["content"] if chat else ""
        return system_parts, history, latest

    def _make_config(self, system_parts: List[str]):
        from google.genai import types  # type: ignore
        config_kwargs = {}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)
        return types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    def complete(self, model: str, messages: List[dict], **kwargs) -> ProviderResult:
        client = self._ensure_client()
        system_parts, history, latest = self._prepare_messages(messages)
        cfg = self._make_config(system_parts)

        chat_session = client.chats.create(model=model, history=history, config=cfg)
        resp = chat_session.send_message(latest)

        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", None) or estimate_messages_tokens(messages)
        out_tok = getattr(usage, "candidates_token_count", None) or estimate_tokens(text)
        return ProviderResult(text=text, model=model, input_tokens=in_tok, output_tokens=out_tok)

    def stream(
        self, model: str, messages: List[dict], **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """Stream response from Gemini using generate_content with stream=True."""
        client = self._ensure_client()
        system_parts, history, latest = self._prepare_messages(messages)

        # Build full contents list for streaming
        contents = list(history)
        contents.append({"role": "user", "parts": [{"text": latest}]})

        config_kwargs = {}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)

        from google.genai import types  # type: ignore
        cfg = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        full_text = []
        try:
            for chunk in client.models.generate_content_stream(
                model=model, contents=contents, config=cfg
            ):
                text = getattr(chunk, "text", "") or ""
                if text:
                    full_text.append(text)
                    yield StreamChunk(text=text, finished=False)
        except (RuntimeError, ValueError):
            raise
        except Exception as stream_err:
            err_msg = str(stream_err).lower()
            if "auth" in err_msg or "api key" in err_msg or "permission" in err_msg:
                raise
            result = self.complete(model, messages, **kwargs)
            yield StreamChunk(
                text=result.text,
                finished=True,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
            return

        combined = "".join(full_text)
        yield StreamChunk(
            text="",
            finished=True,
            input_tokens=estimate_messages_tokens(messages),
            output_tokens=estimate_tokens(combined),
        )
