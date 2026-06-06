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
    start_calls = []
    model_calls = []

    class FakeProvider:
        name = "ollama"

        def is_available(self) -> bool:
            return False

    decision = SimpleNamespace(provider=FakeProvider(), model="llama3.1:8b")
    cfg = Config(mode="local", ollama_host="http://localhost:11434")

    monkeypatch.setattr("memory_router.cli.ensure_ollama_running", lambda host: start_calls.append(host) or True)
    monkeypatch.setattr(
        "memory_router.cli.ensure_ollama_model_available",
        lambda host, model, progress_callback=None: model_calls.append((host, model)) or False,
    )

    _ensure_local_ollama_ready(cfg, decision, force_local=False)

    assert start_calls == ["http://localhost:11434"]
    assert model_calls == [("http://localhost:11434", "llama3.1:8b")]


def test_cli_local_mode_pulls_model_when_ollama_is_already_running(monkeypatch):
    model_calls = []

    class FakeProvider:
        name = "ollama"

        def is_available(self) -> bool:
            return True

    decision = SimpleNamespace(provider=FakeProvider(), model="qwen2.5:14b")
    cfg = Config(mode="local", ollama_host="http://localhost:11434")

    monkeypatch.setattr(
        "memory_router.cli.ensure_ollama_model_available",
        lambda host, model, progress_callback=None: model_calls.append((host, model)) or True,
    )

    _ensure_local_ollama_ready(cfg, decision, force_local=False)

    assert model_calls == [("http://localhost:11434", "qwen2.5:14b")]


def test_ensure_ollama_model_available_pulls_missing_model(monkeypatch):
    pulls = []
    listed = {"before": True}

    def fake_get(url, timeout):
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                if listed["before"]:
                    listed["before"] = False
                    return {"models": [{"name": "llama3.1:8b"}]}
                return {"models": [{"name": "qwen2.5:14b"}]}

        return Response()

    def fake_post(url, json, timeout, stream):
        pulls.append((url, json["name"]))
        assert stream is True

        class Response:
            def raise_for_status(self):
                pass

            def iter_lines(self):
                return [
                    b'{"status":"pulling manifest"}',
                    b'{"status":"pulling layer","completed":10,"total":20}',
                    b'{"status":"success"}',
                ]

        return Response()

    monkeypatch.setattr(ollama_utils.requests, "get", fake_get)
    monkeypatch.setattr(ollama_utils.requests, "post", fake_post)

    pulled = ollama_utils.ensure_ollama_model_available(
        "http://localhost:11434",
        "qwen2.5:14b",
        timeout=1,
    )

    assert pulled is True
    assert pulls == [("http://localhost:11434/api/pull", "qwen2.5:14b")]
