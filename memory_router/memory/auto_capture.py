"""Automatic promotion of useful chat turns into Memory Palace entries.

The CLI already saves conversation history locally. This module decides when a
completed turn is worth keeping as a structured long-term memory and writes a
compact note into the memories database.
"""

from __future__ import annotations

import re
from typing import Optional

from ..classifier import Classification
from ..config import Config
from .sqlite_store import Memory, MemoryStore

_MEMORY_CUE = re.compile(
    r"\b(remember|prefer|always|never|don't forget|do not forget|"
    r"keep in mind|my name is|call me|i like|i want|note that)\b",
    re.IGNORECASE,
)

_CODING_CUE = re.compile(
    r"\b(?:use|prefer|always|never|project uses|repo uses|stack|framework|"
    r"language|version|test with|lint with|format with|build with|pytest|ruff|"
    r"mypy|eslint|prettier|typescript|python|docker|sql|api)\b",
    re.IGNORECASE,
)

_TASK_BONUS = {
    "code": 0.15,
    "security": 0.2,
    "reasoning": 0.15,
    "agentic": 0.2,
    "explain": 0.1,
    "summarize": 0.05,
    "rewrite": 0.05,
}

_MIN_IMPORTANCE_TO_STORE = 0.5
_SENSITIVE_PATTERNS = [
    re.compile(r"\b(?:password|api\s*key|secret|token|bearer|credential|private\s*key)\b", re.I),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----", re.I),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    # Base64 detection: require high entropy (mixed case + digits + special chars)
    # to avoid false positives on normal English sentences.
    re.compile(r"\b[A-Za-z0-9/+]{40,}={1,2}\b"),
]

_INJECTION_PATTERNS = [
    re.compile(r"\bignore (?:all|any|the)?\s*previous instructions\b", re.I),
    re.compile(r"\bdisregard (?:all|any|the)?\s*previous instructions\b", re.I),
    re.compile(r"\b(?:system prompt|developer message|developer instructions|hidden prompt)\b", re.I),
    re.compile(r"\b(?:prompt injection|jailbreak)\b", re.I),
    re.compile(r"\byou are chatgpt\b", re.I),
]

_CODE_HEAVY_PATTERNS = [
    re.compile(r"```"),
    re.compile(r"\bTraceback \(most recent call last\):"),
    re.compile(r"\b(?:SyntaxError|TypeError|ValueError|NameError|ReferenceError|Exception):"),
]


def capture_turn(
    query: str,
    answer: str,
    classification: Classification,
    cfg: Config,
    store: MemoryStore,
    allow_capture: bool = True,
) -> Optional[int]:
    """Promote a completed turn to the Memory Palace when it looks useful.

    Returns the inserted memory id, or ``None`` when the turn should not be
    captured.
    """
    if not allow_capture or not cfg.memory_enabled or not cfg.auto_capture_memories:
        return None

    query = (query or "").strip()
    answer = (answer or "").strip()
    if not query or not answer or answer.lower() == "[no response]":
        return None
    if _looks_sensitive(query) or _looks_sensitive(answer):
        return None
    if _looks_prompt_injection(query) or _looks_prompt_injection(answer):
        return None
    if _looks_code_heavy(query) or _looks_code_heavy(answer):
        return None

    importance = _estimate_importance(query, answer, classification)
    if importance < _MIN_IMPORTANCE_TO_STORE and not _MEMORY_CUE.search(query):
        return None

    content = _build_content(
        query,
        answer,
        coding=classification.task == "code" or classification.domain == "software",
    )
    if store.has_content(content):
        return None

    memory = Memory(
        task=classification.task,
        domain=classification.domain,
        concepts=classification.concepts,
        content=content,
        importance=importance,
        memory_type="episodic",
        source="auto_capture",
    )
    return store.add(memory)


def _build_content(
    query: str,
    answer: str,
    max_query_chars: int = 180,
    max_answer_chars: int = 260,
    coding: bool = False,
) -> str:
    query_excerpt = _compact(query, max_query_chars)
    answer_excerpt = _summarize_answer(answer, max_answer_chars)
    prefix = "Coding note" if coding else "User asked"
    if answer_excerpt:
        return f"{prefix}: {query_excerpt}\nAssistant answer: {answer_excerpt}"
    return f"{prefix}: {query_excerpt}"


def _summarize_answer(text: str, max_chars: int = 260) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    picked = []
    for sentence in sentences:
        if sentence:
            picked.append(sentence.strip())
        if len(picked) >= 2:
            break

    summary = " ".join(picked).strip() or cleaned
    return _compact(summary, max_chars)


def _estimate_importance(query: str, answer: str, classification: Classification) -> float:
    score = 0.35
    score += min(0.35, classification.complexity * 0.35)
    score += _TASK_BONUS.get(classification.task, 0.0)
    if classification.domain != "general":
        score += 0.05
    if _MEMORY_CUE.search(query):
        score += 0.2
    if classification.task == "code" or classification.domain == "software":
        score += 0.1
        if _CODING_CUE.search(query) or _CODING_CUE.search(answer):
            score += 0.15
    if len(answer) > 400:
        score += 0.05
    return max(0.0, min(1.0, score))


def _compact(text: str, max_chars: int) -> str:
    cleaned = " ".join((text or "").strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _looks_sensitive(text: str) -> bool:
    compacted = " ".join((text or "").split())
    if not compacted:
        return False
    return any(pattern.search(compacted) for pattern in _SENSITIVE_PATTERNS)


def _looks_prompt_injection(text: str) -> bool:
    compacted = " ".join((text or "").split())
    if not compacted:
        return False
    return any(pattern.search(compacted) for pattern in _INJECTION_PATTERNS)


def _looks_code_heavy(text: str) -> bool:
    compacted = (text or "").strip()
    if not compacted:
        return False
    return any(pattern.search(compacted) for pattern in _CODE_HEAVY_PATTERNS)
