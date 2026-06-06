from __future__ import annotations

from types import SimpleNamespace

from memory_router.cli import _ensure_local_ollama_ready
from memory_router.config import Config
from memory_router.utils import ollama as ollama_utils


def test_ensure_ollama_running_starts_background_server(monkeypatch):
    started = {}

    class FakeProvider:
        def __init__(self, host: str):
            self.host = host
            self.calls = 0

        def is_available(self) -> bool:
            self.calls += 1
            return self.calls >= 3

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        started["cmd"] = cmd
        started["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(ollama_utils, "OllamaProvider", FakeProvider)
    monkeypatch.setattr(ollama_utils.shutil, "which", lambda name: "/usr/local/bin/ollama")
    monkeypatch.setattr(ollama_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ollama_utils.time, "sleep", lambda _: None)

    assert ollama_utils.ensure_ollama_running("http://localhost:11434", timeout=1) is True
    assert started["cmd"] == ["ollama", "serve"]
    assert started["kwargs"]["env"]["OLLAMA_HOST"] == "localhost:11434"


def test_cli_local_mode_triggers_ollama_autostart(monkeypatch):
    calls = []

    class FakeProvider:
        name = "ollama"

        def is_available(self) -> bool:
            return False

    decision = SimpleNamespace(provider=FakeProvider())
    cfg = Config(mode="local", ollama_host="http://localhost:11434")

    monkeypatch.setattr("memory_router.cli.ensure_ollama_running", lambda host: calls.append(host) or True)

    _ensure_local_ollama_ready(cfg, decision, force_local=False)

    assert calls == ["http://localhost:11434"]
