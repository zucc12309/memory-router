"""Token estimation utilities.

Uses tiktoken when available for accurate counts; falls back to an improved
heuristic that differentiates code vs prose (code has shorter tokens on average
due to operators, brackets, and short variable names).
"""

from __future__ import annotations

from typing import Iterable, Optional

# Lazy-loaded tiktoken encoding
_ENCODING = None
_TIKTOKEN_AVAILABLE: Optional[bool] = None


def _get_encoding():
    """Load tiktoken encoding lazily, caching the result."""
    global _ENCODING, _TIKTOKEN_AVAILABLE
    if _TIKTOKEN_AVAILABLE is False:
        return None
    if _ENCODING is not None:
        return _ENCODING
    try:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("cl100k_base")
        _TIKTOKEN_AVAILABLE = True
        return _ENCODING
    except (ImportError, Exception):
        _TIKTOKEN_AVAILABLE = False
        return None


def estimate_tokens(text: str) -> int:
    """Token count — uses tiktoken if available, improved heuristic otherwise."""
    if not text:
        return 0

    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(text))

    # Improved heuristic: different ratio for code vs prose
    chars = len(text)
    code_indicators = (
        text.count("{")
        + text.count("}")
        + text.count("(")
        + text.count(")")
        + text.count(";")
        + text.count("=>")
    )
    if code_indicators > chars / 50:
        return max(1, int(chars / 3.2))  # code is ~3.2 chars/token
    return max(1, int(chars / 3.7))  # prose is ~3.7 chars/token


def estimate_messages_tokens(messages: Iterable[dict]) -> int:
    """Estimate tokens for a list of {role, content} messages."""
    total = 0
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else str(m)
        total += estimate_tokens(content) + 4  # small overhead per message
    return total


def percent_saved(full_tokens: int, sent_tokens: int) -> int:
    """Percentage of tokens saved by sending the trimmed context vs the full history."""
    if full_tokens <= 0:
        return 0
    saved = max(0, full_tokens - sent_tokens)
    return int(round(100 * saved / full_tokens))


# ---------- pricing table ----------
# Approximate USD per 1M tokens. Update as providers change pricing.
# Format: model_id_prefix -> (input_per_million, output_per_million)
_PRICING = {
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o3": (10.00, 40.00),
    "o4-mini": (1.10, 4.40),
    # Anthropic
    "claude-haiku": (0.80, 4.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (15.00, 75.00),
    # Google
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    # Local — free
    "llama": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "qwen": (0.0, 0.0),
    "phi": (0.0, 0.0),
    "deepseek": (0.0, 0.0),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Best-effort cost estimate. Returns 0.0 if model isn't in the table."""
    m = (model or "").lower()
    rate_in, rate_out = 0.0, 0.0
    # Match the longest prefix so 'claude-opus-4-7' lands on 'claude-opus'.
    best = ""
    for key in _PRICING:
        if m.startswith(key) or key in m:
            if len(key) > len(best):
                best = key
    if best:
        rate_in, rate_out = _PRICING[best]
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


def format_cost(usd: float) -> str:
    if usd <= 0:
        return "free (local)"
    if usd < 0.0001:
        return "<$0.0001"
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.3f}"
