"""System-spec helpers for Ollama recommendations."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SystemSpecs:
    """A small summary of the current machine."""

    os_name: str
    architecture: str
    cpu_count: int
    memory_gb: Optional[float]


@dataclass(frozen=True)
class OllamaRecommendation:
    """A model recommendation with a short explanation."""

    model: str
    reason: str
    tier: str


@dataclass(frozen=True)
class OllamaModelOption:
    """One candidate model in the local Ollama shortlist."""

    model: str
    min_memory_gb: float
    label: str
    description: str


OLLAMA_MODEL_OPTIONS: List[OllamaModelOption] = [
    OllamaModelOption(
        model="llama3.2:3b",
        min_memory_gb=4.0,
        label="Fastest",
        description="Best when you want the lightest local model.",
    ),
    OllamaModelOption(
        model="llama3.1:8b",
        min_memory_gb=8.0,
        label="Balanced",
        description="Good default for most laptops and desktops.",
    ),
    OllamaModelOption(
        model="qwen2.5:14b",
        min_memory_gb=16.0,
        label="Stronger",
        description="Better coding and reasoning if you have the RAM.",
    ),
    OllamaModelOption(
        model="qwen2.5:32b",
        min_memory_gb=32.0,
        label="Large",
        description="Higher quality on machines with substantial memory.",
    ),
    OllamaModelOption(
        model="llama3.1:70b",
        min_memory_gb=64.0,
        label="Workstation",
        description="Only for very large-memory systems.",
    ),
]

_CONFIRMATION_WORDS = {"y", "yes", "n", "no", "true", "false", "on", "off"}


def detect_system_specs() -> SystemSpecs:
    """Detect the current machine's broad specs using stdlib only."""
    memory_gb = _detect_memory_gb()
    return SystemSpecs(
        os_name=platform.system() or "unknown",
        architecture=platform.machine() or platform.processor() or "unknown",
        cpu_count=os.cpu_count() or 1,
        memory_gb=memory_gb,
    )


def recommend_ollama_model(specs: SystemSpecs) -> OllamaRecommendation:
    """Choose a sane local Ollama model based on RAM."""
    memory_gb = specs.memory_gb
    if memory_gb is None:
        return OllamaRecommendation(
            model="llama3.1:8b",
            reason="RAM could not be detected, so a balanced 8B model is the safest default.",
            tier="balanced",
        )

    chosen = OLLAMA_MODEL_OPTIONS[0]
    for option in OLLAMA_MODEL_OPTIONS:
        if memory_gb >= option.min_memory_gb:
            chosen = option

    reason = (
        f"Detected about {memory_gb:.1f} GB RAM on {specs.architecture}; "
        f"{chosen.model} is the best fit from the local shortlist."
    )
    return OllamaRecommendation(
        model=chosen.model,
        reason=reason,
        tier=chosen.label.lower(),
    )


def is_valid_ollama_model_name(model: str) -> bool:
    """Return False for blank values and common mistaken yes/no answers."""
    value = (model or "").strip()
    if not value:
        return False
    if value.lower() in _CONFIRMATION_WORDS:
        return False
    if any(ch.isspace() for ch in value):
        return False
    return True


def normalize_ollama_model_name(model: str, fallback: str = "") -> str:
    """Return a usable Ollama model id, or fallback when input is clearly invalid."""
    value = (model or "").strip()
    return value if is_valid_ollama_model_name(value) else fallback


def _detect_memory_gb() -> Optional[float]:
    """Try to estimate installed memory using portable stdlib APIs."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages and page_size:
            return round((pages * page_size) / (1024**3), 1)
    except (AttributeError, ValueError, OSError):
        pass

    return None
