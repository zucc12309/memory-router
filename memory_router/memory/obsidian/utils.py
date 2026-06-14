"""Pure helpers for the Obsidian layer: filenames, paths, time, redaction.

Standard library only. No I/O here — these are deterministic transforms so
they are trivial to unit test.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Filenames & slugs
# ---------------------------------------------------------------------------

_ILLEGAL_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WS = re.compile(r"\s+")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """Make a string safe to use as a single path component.

    Strips path separators, illegal filesystem characters, and leading dots
    (so a memory can never produce ``../`` or a dotfile). Never returns "".
    """
    name = (name or "").strip()
    # Kill path-traversal and separators outright.
    name = name.replace("..", "").replace("/", "-").replace("\\", "-")
    name = _ILLEGAL_FS.sub("", name)
    name = _WS.sub(" ", name).strip(" .")
    if len(name) > max_len:
        name = name[:max_len].rstrip(" .-")
    return name or "untitled"


def slugify(text: str, max_len: int = 60) -> str:
    """Lowercase hyphen-slug for wikilink-friendly identifiers."""
    slug = _SLUG_STRIP.sub("-", (text or "").lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "untitled"


def safe_join(base: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` under ``base``, refusing escapes.

    Raises ValueError if the resolved path would land outside ``base`` — the
    core defense against path traversal from attacker-influenced titles.
    """
    base_resolved = base.resolve()
    candidate = (base_resolved / rel_path).resolve()
    if base_resolved != candidate and base_resolved not in candidate.parents:
        raise ValueError(f"Unsafe path escapes vault: {rel_path!r}")
    return candidate


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def to_iso(epoch: float) -> str:
    """Epoch seconds → ISO-8601 UTC (e.g. 2026-06-14T10:30:00Z)."""
    if not epoch:
        return ""
    dt = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def from_iso(value: str) -> float:
    """ISO-8601 (or bare epoch) → epoch seconds. Returns 0.0 on failure."""
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)  # already an epoch string
    except ValueError:
        pass
    try:
        text = text.replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

_PLACEHOLDER = "[REDACTED]"

# Order matters: more specific patterns first. Each entry is (label, regex).
_SECRET_PATTERNS = [
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("google_key", re.compile(r"\bAIza[A-Za-z0-9_\-]{30,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    (
        "assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key)\b"
            r"\s*[:=]\s*[\"']?[^\s\"']{6,}"
        ),
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
]


def redact(text: str) -> Tuple[str, int]:
    """Strip secrets/PII from ``text``. Returns (clean_text, n_redactions).

    One-way by design: the original always survives in SQLite. For assignment
    patterns we keep the key name and replace only the value.
    """
    if not text:
        return text, 0
    count = 0
    out = text
    for label, pattern in _SECRET_PATTERNS:
        if label == "assignment":
            def _repl(m: "re.Match[str]") -> str:
                nonlocal count
                count += 1
                key = re.split(r"[:=]", m.group(0), maxsplit=1)[0]
                return f"{key}={_PLACEHOLDER}"

            out = pattern.sub(_repl, out)
        else:
            out, n = pattern.subn(_PLACEHOLDER, out)
            count += n
    return out, count
