"""Lightweight rule-based classifier.

Classifies a user query into (task, domain, concepts) so the router can pick
a model and the memory palace can pull relevant memories. This is intentionally
simple — no LLM call — to keep the pre-routing path fast and free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# (regex, label) pairs. Order matters: first match wins.
_TASK_RULES = [
    (r"\b(write|implement|refactor|fix|debug|patch)\b.*\b(code|function|bug|class|script)\b", "code"),
    (r"\b(security|vulnerab|exploit|cve|owasp|injection)\b", "security"),
    (r"\b(prove|derive|theorem|lemma|integrate|differentiate|solve)\b", "reasoning"),
    (r"\b(plan|design|architect|orchestrate|workflow|multi-?step)\b", "agentic"),
    (r"\b(explain|what is|why|how does|describe|teach)\b", "explain"),
    (r"\b(summari[sz]e|tl;dr|recap)\b", "summarize"),
    (r"\b(translate|rewrite|paraphrase|rephrase)\b", "rewrite"),
]

_DOMAIN_RULES = [
    (r"\b(stock|bond|yield|coupon|duration|convexity|portfolio|equity|finance|nav|cagr)\b", "finance"),
    (r"\b(python|javascript|typescript|react|sql|api|docker|kubernetes|git|regex)\b", "software"),
    (r"\b(neural|ml|model|training|gradient|tensor|pytorch|llm|transformer)\b", "ml"),
    (r"\b(law|legal|contract|gdpr|hipaa|compliance)\b", "legal"),
    (r"\b(medical|disease|symptom|diagnosis|drug|clinical)\b", "medical"),
    (r"\b(physics|chemistry|biology|geology|astronomy)\b", "science"),
]

_STOP = {
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "are",
    "what", "how", "why", "when", "where", "explain", "tell", "me", "i", "you",
    "do", "does", "with", "about", "this", "that", "it", "be", "by", "as",
    "can", "please", "would", "should", "could",
}


@dataclass
class Classification:
    task: str
    domain: str
    concepts: List[str]
    complexity: float  # 0.0 - 1.0; routes to bigger models when high

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "domain": self.domain,
            "concepts": self.concepts,
            "complexity": self.complexity,
        }


def classify(query: str) -> Classification:
    q = (query or "").lower()
    task = _first_match(q, _TASK_RULES, default="general")
    domain = _first_match(q, _DOMAIN_RULES, default="general")
    concepts = _extract_concepts(q)
    complexity = _estimate_complexity(q, task)
    return Classification(task=task, domain=domain, concepts=concepts, complexity=complexity)


def _first_match(text: str, rules, default: str) -> str:
    for pattern, label in rules:
        if re.search(pattern, text):
            return label
    return default


def _extract_concepts(text: str, limit: int = 5) -> List[str]:
    """Pull salient noun-ish tokens. Naive but useful for retrieval boosting."""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", text.lower())
    seen = []
    for w in words:
        if w in _STOP or w in seen:
            continue
        seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def _estimate_complexity(text: str, task: str) -> float:
    score = 0.0
    # Long queries tend to need more capable models.
    score += min(0.4, len(text) / 1000)
    # Task-driven priors.
    score += {
        "code": 0.4,
        "security": 0.6,
        "reasoning": 0.5,
        "agentic": 0.7,
        "explain": 0.2,
        "summarize": 0.1,
        "rewrite": 0.1,
        "general": 0.2,
    }.get(task, 0.2)
    # Keywords that suggest depth.
    if re.search(r"\b(prove|derive|architecture|design|threat model|optimi[sz]e)\b", text):
        score += 0.2
    return max(0.0, min(1.0, score))
