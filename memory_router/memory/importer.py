"""Memory import/export for ChatGPT, Claude, and generic JSON formats.

Supports:
  - ChatGPT conversations.json export
  - Claude conversations JSON export
  - Generic JSON: [{content, domain?, task?, importance?, concepts?}, ...]
  - Memory Router native export/import (round-trip)

All imports deduplicate by content to avoid duplicate memories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .sqlite_store import Memory, MemoryStore


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_memories(
    store: MemoryStore,
    limit: int = 100_000,
) -> List[Dict[str, Any]]:
    """Export all memories as a JSON-serialisable list."""
    mems = store.list_all(limit=limit)
    return [
        {
            "id": m.id,
            "content": m.content,
            "domain": m.domain,
            "task": m.task,
            "concepts": m.concepts,
            "importance": m.importance,
            "confidence": m.confidence,
            "memory_type": m.memory_type,
            "source": m.source,
            "created_at": m.created_at,
            "last_used": m.last_used,
            "usage_count": m.usage_count,
        }
        for m in mems
    ]


def export_to_file(store: MemoryStore, path: Path, encrypt: Optional[bool] = None) -> int:
    """Export memories to a JSON file. Returns count exported.

    When encrypt is None (default), encryption is used if encryption_enabled
    is set in config and the cryptography package is available.
    When encrypt=True, the file is AES-256-GCM encrypted using the machine key.
    The encrypted file has a .enc extension hint and cannot be read on other machines.
    """
    if encrypt is None:
        from ..config import load_config
        from ..security.encryption import is_encryption_available
        cfg = load_config()
        encrypt = cfg.encryption_enabled and is_encryption_available()
    data = export_memories(store)
    payload = json.dumps(
        {"format": "memory-router", "version": 2, "memories": data},
        indent=2, ensure_ascii=False,
    )

    if encrypt:
        from ..security.encryption import encrypt_content, is_encryption_available

        if not is_encryption_available():
            raise RuntimeError(
                "Encryption requires the cryptography package. "
                "Install: pip install memory-router[encryption]"
            )
        encrypted = encrypt_content(payload)
        path.write_bytes(b"MR_ENC\x01" + encrypted)  # Magic header + version
    else:
        path.write_text(payload, encoding="utf-8")

    return len(data)


# ---------------------------------------------------------------------------
# Import — native Memory Router format
# ---------------------------------------------------------------------------

def import_from_file(
    store: MemoryStore,
    path: Path,
    source: str = "import",
) -> Tuple[int, int]:
    """Import memories from a JSON file.

    Auto-detects format (Memory Router native, ChatGPT, Claude, generic list).
    Returns (imported_count, skipped_duplicates).
    """
    file_bytes = path.read_bytes()

    # Check for encrypted export
    if file_bytes[:7] == b"MR_ENC\x01":
        from ..security.encryption import decrypt_content, is_encryption_available

        if not is_encryption_available():
            raise RuntimeError(
                "This file is encrypted. Install: pip install memory-router[encryption]"
            )
        plaintext = decrypt_content(file_bytes[7:])
        raw = json.loads(plaintext)
    else:
        raw = json.loads(file_bytes.decode("utf-8"))

    if isinstance(raw, dict):
        fmt = raw.get("format", "")
        if fmt == "memory-router":
            return _import_native(store, raw.get("memories", []), source)
        if "mapping" in raw or "conversations" in raw:
            return _import_chatgpt(store, raw, source)
        if isinstance(raw.get("memories"), list):
            return _import_native(store, raw["memories"], source)
        # Single conversation object — try Claude
        if "uuid" in raw or "chat_messages" in raw:
            return _import_claude(store, [raw], source)
        # Fallback: treat top-level dict values as memories
        return _import_generic(store, list(raw.values()), source)

    if isinstance(raw, list):
        # Heuristic: if first item has "mapping" or "title", it's ChatGPT
        if raw and isinstance(raw[0], dict):
            if "mapping" in raw[0]:
                return _import_chatgpt(store, {"conversations": raw}, source)
            if "uuid" in raw[0] or "chat_messages" in raw[0]:
                return _import_claude(store, raw, source)
        return _import_generic(store, raw, source)

    return 0, 0


def _import_native(
    store: MemoryStore, items: List[dict], source: str
) -> Tuple[int, int]:
    """Import Memory Router native format."""
    imported, skipped = 0, 0
    for item in items:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if store.has_content(content):
            skipped += 1
            continue
        store.add(Memory(
            content=content,
            domain=item.get("domain", "general"),
            task=item.get("task", "general"),
            concepts=item.get("concepts", []),
            importance=float(item.get("importance", 0.5)),
            confidence=float(item.get("confidence", 1.0)),
            memory_type=item.get("memory_type", "semantic"),
            source=source,
        ))
        imported += 1
    return imported, skipped


# ---------------------------------------------------------------------------
# Import — ChatGPT conversations.json
# ---------------------------------------------------------------------------

def _import_chatgpt(
    store: MemoryStore, data: dict, source: str
) -> Tuple[int, int]:
    """Extract user messages from ChatGPT's conversations.json export.

    ChatGPT exports look like:
    [
      {
        "title": "...",
        "mapping": {
          "<uuid>": {
            "message": {
              "author": {"role": "user"|"assistant"|"system"},
              "content": {"parts": ["..."]}
            }
          }
        }
      }
    ]
    """
    conversations = data.get("conversations") or data.get("mapping")
    if isinstance(conversations, dict):
        conversations = [{"mapping": conversations}]
    if not isinstance(conversations, list):
        return 0, 0

    imported, skipped = 0, 0
    for conv in conversations:
        mapping = conv.get("mapping", {})
        if not isinstance(mapping, dict):
            continue

        for node_id, node in mapping.items():
            msg = (node.get("message") or {})
            author = (msg.get("author") or {}).get("role", "")
            if author != "user":
                continue

            content_obj = msg.get("content", {})
            parts = content_obj.get("parts", []) if isinstance(content_obj, dict) else []
            text = " ".join(str(p) for p in parts if isinstance(p, str)).strip()

            if not text or len(text) < 20:
                continue
            if len(text) > 500:
                text = text[:497] + "..."
            if store.has_content(text):
                skipped += 1
                continue

            store.add(Memory(
                content=text,
                domain="general",
                task="general",
                concepts=_extract_concepts_simple(text),
                importance=0.4,
                memory_type="episodic",
                source=source,
            ))
            imported += 1

    return imported, skipped


# ---------------------------------------------------------------------------
# Import — Claude conversation export
# ---------------------------------------------------------------------------

def _import_claude(
    store: MemoryStore, conversations: list, source: str
) -> Tuple[int, int]:
    """Extract user messages from a Claude conversation export.

    Claude exports vary, but common shapes:
    - {"uuid": "...", "chat_messages": [{"sender": "human", "text": "..."}]}
    - [{"role": "user", "content": "..."}]
    """
    imported, skipped = 0, 0

    for conv in conversations:
        messages = conv.get("chat_messages", [])
        if not messages and isinstance(conv, dict):
            # Maybe it's a flat message list
            if "role" in conv or "sender" in conv:
                messages = [conv]

        for msg in messages:
            sender = msg.get("sender", msg.get("role", ""))
            if sender not in ("human", "user"):
                continue

            text = (msg.get("text") or msg.get("content") or "").strip()
            if not text or len(text) < 20:
                continue
            if len(text) > 500:
                text = text[:497] + "..."
            if store.has_content(text):
                skipped += 1
                continue

            store.add(Memory(
                content=text,
                domain="general",
                task="general",
                concepts=_extract_concepts_simple(text),
                importance=0.4,
                memory_type="episodic",
                source=source,
            ))
            imported += 1

    return imported, skipped


# ---------------------------------------------------------------------------
# Import — generic list of strings or objects
# ---------------------------------------------------------------------------

def _import_generic(
    store: MemoryStore, items: list, source: str
) -> Tuple[int, int]:
    """Import a flat list of strings or {content:...} objects."""
    imported, skipped = 0, 0

    for item in items:
        if isinstance(item, str):
            content = item.strip()
            domain = "general"
            task = "general"
            concepts = []
            importance = 0.5
        elif isinstance(item, dict):
            content = (item.get("content") or item.get("text") or "").strip()
            domain = item.get("domain", "general")
            task = item.get("task", "general")
            concepts = item.get("concepts", [])
            importance = float(item.get("importance", 0.5))
        else:
            continue

        if not content or len(content) < 5:
            continue
        if store.has_content(content):
            skipped += 1
            continue

        store.add(Memory(
            content=content,
            domain=domain,
            task=task,
            concepts=concepts if isinstance(concepts, list) else [],
            importance=importance,
            memory_type="semantic",
            source=source,
        ))
        imported += 1

    return imported, skipped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "are",
    "what", "how", "why", "when", "where", "this", "that", "it", "be", "by",
    "can", "please", "would", "should", "could", "do", "does", "with", "about",
    "me", "i", "you", "my", "your", "we", "they", "he", "she", "at", "from",
}


def _extract_concepts_simple(text: str, limit: int = 5) -> List[str]:
    """Quick keyword extraction for imported memories."""
    import re
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", text.lower())
    seen: List[str] = []
    for w in words:
        if w in _STOP or w in seen:
            continue
        seen.append(w)
        if len(seen) >= limit:
            break
    return seen
