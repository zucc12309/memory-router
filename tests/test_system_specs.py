from __future__ import annotations

from memory_router.utils.system import detect_system_specs, recommend_ollama_model


def test_recommend_ollama_model_uses_detected_ram(monkeypatch):
    monkeypatch.setattr("memory_router.utils.system.platform.system", lambda: "Darwin")
    monkeypatch.setattr("memory_router.utils.system.platform.machine", lambda: "arm64")
    monkeypatch.setattr("memory_router.utils.system.platform.processor", lambda: "")
    monkeypatch.setattr("memory_router.utils.system.os.cpu_count", lambda: 8)
    monkeypatch.setattr(
        "memory_router.utils.system.os.sysconf",
        lambda key: {
            "SC_PHYS_PAGES": 4_194_304,
            "SC_PAGE_SIZE": 4096,
        }[key],
    )

    specs = detect_system_specs()
    recommendation = recommend_ollama_model(specs)

    assert specs.os_name == "Darwin"
    assert specs.architecture == "arm64"
    assert specs.cpu_count == 8
    assert specs.memory_gb == 16.0
    assert recommendation.model == "qwen2.5:14b"
    assert "16.0 GB" in recommendation.reason


def test_recommend_ollama_model_falls_back_when_memory_unknown(monkeypatch):
    monkeypatch.setattr("memory_router.utils.system.platform.system", lambda: "Linux")
    monkeypatch.setattr("memory_router.utils.system.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("memory_router.utils.system.platform.processor", lambda: "")
    monkeypatch.setattr("memory_router.utils.system.os.cpu_count", lambda: 4)
    monkeypatch.setattr(
        "memory_router.utils.system.os.sysconf",
        lambda key: (_ for _ in ()).throw(OSError("unsupported")),
    )

    specs = detect_system_specs()
    recommendation = recommend_ollama_model(specs)

    assert specs.memory_gb is None
    assert recommendation.model == "llama3.1:8b"
