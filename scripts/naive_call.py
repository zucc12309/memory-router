"""Call Gemini directly with the FULL conversation history — no Memory Router.

This is the apples-to-apples baseline: send everything a naive chat client
would have sent, then print the real input/output tokens Google billed.
Compare these numbers against the Memory Router run for the same query.

Usage:
    source ~/Documents/GitHub/memory-router/.venv/bin/activate
    python scripts/naive_call.py "Write a Dockerfile for it"
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import keyring
from google import genai


SERVICE = "memory-router"
DB = Path.home() / ".memory-router" / "conversations.sqlite"
SESSION = "default"
MODEL = "gemini-2.5-flash"
PRICE_IN_PER_M = 0.10   # gemini-2.5-flash input USD per 1M tokens
PRICE_OUT_PER_M = 0.40  # gemini-2.5-flash output USD per 1M tokens


def load_history():
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
        (SESSION,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def to_gemini(history, new_query):
    contents = []
    for m in history:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    contents.append({"role": "user", "parts": [{"text": new_query}]})
    return contents


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python scripts/naive_call.py "your query here"')
    query = " ".join(sys.argv[1:])

    api_key = keyring.get_password(SERVICE, "gemini")
    if not api_key:
        raise SystemExit("No Gemini key in keyring. Run: memory-router auth gemini")

    history = load_history()
    print(f"Loaded {len(history)} prior messages from session '{SESSION}'.")
    print(f"Sending FULL history + new query to {MODEL} (no Memory Router)...\n")

    contents = to_gemini(history, query)
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=MODEL, contents=contents)

    usage = resp.usage_metadata
    in_tok = usage.prompt_token_count
    out_tok = usage.candidates_token_count
    cost = (in_tok * PRICE_IN_PER_M + out_tok * PRICE_OUT_PER_M) / 1_000_000

    text = (resp.text or "").strip()
    preview = text[:300] + ("..." if len(text) > 300 else "")

    print("=== Answer (truncated to 300 chars) ===")
    print(preview)
    print()
    print("=== Naive token usage (no Memory Router) ===")
    print(f"  Input tokens (real):  {in_tok:,}")
    print(f"  Output tokens (real): {out_tok:,}")
    print(f"  Total tokens:         {in_tok + out_tok:,}")
    print(f"  Cost (estimate):      ${cost:.4f}")
    print()
    print("Compare 'Input tokens (real)' above against what Memory Router")
    print("printed for the same query in its Token usage panel.")


if __name__ == "__main__":
    main()
