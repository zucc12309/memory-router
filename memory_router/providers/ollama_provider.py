"""Ollama provider — talks to a local Ollama server over HTTP.

We use plain `requests` so there's no ollama-python dependency. Ollama runs
on http://localhost:11434 by default and supports an OpenAI-style /api/chat
endpoint that takes {role, content} messages.
"""

from __future__ import annotations

from typing import List

import requests

from .base import BaseProvider, ProviderResult
from ..utils.tokens import estimate_tokens, estimate_messages_tokens


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434", timeout: int = 120):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def complete(self, model: str, messages: List[dict], **kwargs) -> ProviderResult:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content", "") or data.get("response", "")
        return ProviderResult(
            text=text,
            model=model,
            input_tokens=estimate_messages_tokens(messages),
            output_tokens=estimate_tokens(text),
        )
