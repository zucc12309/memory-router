"""Cumulative usage statistics.

Every CLI provider call and every MCP `build_context` tool invocation records
a row here. This is what powers `memory-router stats` and the
`stats_summary` MCP tool — so users can see *across all sessions* how many
tokens Memory Router has saved, how many memories got used, and what the
estimated cost impact has been.

The store is intentionally simple — one SQLite table, append-only. No PII,
no prompt content, no answer text. Only the size/source/cost numbers.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import ROOT_DIR, ensure_dirs


STATS_DB = ROOT_DIR / "stats.sqlite"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,                -- 'cli_ask' | 'mcp_build_context' | ...
    naive_tokens INTEGER NOT NULL DEFAULT 0,
    sent_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    memories_used INTEGER NOT NULL DEFAULT 0,
    provider TEXT,
    model TEXT,
    cost_usd REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_events(ts);
CREATE INDEX IF NOT EXISTS idx_usage_kind ON usage_events(kind);
"""


def _connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(str(STATS_DB))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def record_usage(
    kind: str,
    naive_tokens: int = 0,
    sent_tokens: int = 0,
    output_tokens: int = 0,
    memories_used: int = 0,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    cost_usd: float = 0.0,
) -> None:
    """Append a single usage event. Never raises — stats are best-effort."""
    try:
        conn = _connect()
        conn.execute(
            """INSERT INTO usage_events
               (ts, kind, naive_tokens, sent_tokens, output_tokens,
                memories_used, provider, model, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                kind,
                int(naive_tokens or 0),
                int(sent_tokens or 0),
                int(output_tokens or 0),
                int(memories_used or 0),
                provider,
                model,
                float(cost_usd or 0.0),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        # Never let stats writes interfere with the actual user request.
        pass


@dataclass
class StatsSummary:
    calls: int
    naive_tokens: int
    sent_tokens: int
    tokens_saved: int
    saved_pct: int
    output_tokens: int
    memories_used: int
    cost_usd: float
    by_provider: dict
    by_kind: dict

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "naive_tokens": self.naive_tokens,
            "sent_tokens": self.sent_tokens,
            "tokens_saved": self.tokens_saved,
            "saved_pct": self.saved_pct,
            "output_tokens": self.output_tokens,
            "memories_used": self.memories_used,
            "cost_usd": round(self.cost_usd, 6),
            "by_provider": self.by_provider,
            "by_kind": self.by_kind,
        }


def _query(conn, sql_no_filter: str, sql_filtered: str, params: tuple):
    """Run either the unfiltered or filtered SQL based on params."""
    if params:
        return conn.execute(sql_filtered, params)
    return conn.execute(sql_no_filter)


def summarize_stats(since: Optional[float] = None) -> StatsSummary:
    """Aggregate the usage_events table into a single summary."""
    conn = _connect()
    params: tuple = ()
    if since is not None:
        params = (since,)

    row = _query(
        conn,
        """SELECT COUNT(*),
                  COALESCE(SUM(naive_tokens), 0),
                  COALESCE(SUM(sent_tokens), 0),
                  COALESCE(SUM(output_tokens), 0),
                  COALESCE(SUM(memories_used), 0),
                  COALESCE(SUM(cost_usd), 0)
           FROM usage_events""",
        """SELECT COUNT(*),
                  COALESCE(SUM(naive_tokens), 0),
                  COALESCE(SUM(sent_tokens), 0),
                  COALESCE(SUM(output_tokens), 0),
                  COALESCE(SUM(memories_used), 0),
                  COALESCE(SUM(cost_usd), 0)
           FROM usage_events WHERE ts >= ?""",
        params,
    ).fetchone()

    by_provider = {}
    for prov, calls, naive, sent, cost in _query(
        conn,
        """SELECT COALESCE(provider, 'n/a'),
                  COUNT(*),
                  COALESCE(SUM(naive_tokens), 0),
                  COALESCE(SUM(sent_tokens), 0),
                  COALESCE(SUM(cost_usd), 0)
           FROM usage_events
           GROUP BY provider""",
        """SELECT COALESCE(provider, 'n/a'),
                  COUNT(*),
                  COALESCE(SUM(naive_tokens), 0),
                  COALESCE(SUM(sent_tokens), 0),
                  COALESCE(SUM(cost_usd), 0)
           FROM usage_events WHERE ts >= ?
           GROUP BY provider""",
        params,
    ).fetchall():
        by_provider[prov] = {
            "calls": calls,
            "naive_tokens": naive,
            "sent_tokens": sent,
            "cost_usd": round(cost, 6),
        }

    by_kind = {}
    for kind, calls, naive, sent in _query(
        conn,
        """SELECT kind, COUNT(*),
                  COALESCE(SUM(naive_tokens), 0),
                  COALESCE(SUM(sent_tokens), 0)
           FROM usage_events
           GROUP BY kind""",
        """SELECT kind, COUNT(*),
                  COALESCE(SUM(naive_tokens), 0),
                  COALESCE(SUM(sent_tokens), 0)
           FROM usage_events WHERE ts >= ?
           GROUP BY kind""",
        params,
    ).fetchall():
        by_kind[kind] = {
            "calls": calls,
            "naive_tokens": naive,
            "sent_tokens": sent,
        }

    conn.close()

    calls, naive, sent, output, mems, cost = row
    saved = max(0, naive - sent)
    pct = int(round(100 * saved / naive)) if naive > 0 else 0
    return StatsSummary(
        calls=calls,
        naive_tokens=naive,
        sent_tokens=sent,
        tokens_saved=saved,
        saved_pct=pct,
        output_tokens=output,
        memories_used=mems,
        cost_usd=cost,
        by_provider=by_provider,
        by_kind=by_kind,
    )


def reset_stats() -> int:
    """Wipe the usage_events table. Returns rows deleted."""
    conn = _connect()
    cur = conn.execute("DELETE FROM usage_events")
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n
