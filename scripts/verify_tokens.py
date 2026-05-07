"""Verify Memory Router's token-saving claim.

Reads your real conversation log from ~/.memory-router/conversations.sqlite
and asks Google's tokenizer how many tokens a *naive* client would have sent
for the latest query — i.e. the entire chat history concatenated, no trimming,
no memory retrieval, no summary. Compare against the "Input tokens (real)"
that Memory Router showed when you ran the query.

Run from the repo's venv:
    source ~/Documents/GitHub/memory-router/.venv/bin/activate
    python scripts/verify_tokens.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import keyring
from google import genai


SERVICE = "memory-router"
DB = Path.home() / ".memory-router" / "conversations.sqlite"
SESSION = "default"
MODEL = "gemini-2.5-flash"


def load_messages(session_id: str = SESSION):
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def to_gemini(messages):
    """Translate {role: user|assistant, content} -> Gemini's contents shape."""
    out = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        out.append({"role": role, "parts": [{"text": m["content"]}]})
    return out


def main():
    api_key = keyring.get_password(SERVICE, "gemini")
    if not api_key:
        raise SystemExit("No Gemini key found in keyring. Run: memory-router auth gemini")

    msgs = load_messages()
    if not msgs:
        raise SystemExit("No conversation history found. Run a few queries first.")

    print(f"Loaded {len(msgs)} messages from session '{SESSION}'.")
    print(f"  Roles: {sum(1 for m in msgs if m['role']=='user')} user, "
          f"{sum(1 for m in msgs if m['role']=='assistant')} assistant.\n")

    contents = to_gemini(msgs)

    client = genai.Client(api_key=api_key)
    resp = client.models.count_tokens(model=MODEL, contents=contents)
    naive_total = resp.total_tokens
    print(f"Naive baseline (real, from Google tokenizer):")
    print(f"  Full history through model={MODEL}: {naive_total:,} tokens\n")

    # Cost the naive call would have incurred (input only).
    cost_in = naive_total * 0.10 / 1_000_000  # $0.10 per 1M for gemini-2.5-flash input
    print(f"  Cost if sent as-is at gemini-2.5-flash input rates: ${cost_in:.4f}")
    print()
    print("Compare this against the 'Input tokens (real)' Memory Router")
    print("printed for your most recent query. The difference is your real saving.")


if __name__ == "__main__":
    main()
