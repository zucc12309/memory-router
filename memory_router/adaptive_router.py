"""Adaptive router with outcome learning.

Wraps the rule-based Router and records every completion outcome into
route_history.sqlite. Over time, the adaptive router uses historical
quality, cost, and latency data to improve model selection.

The learning is simple and interpretable:
  - For each (provider, model, task) triple, track average quality,
    latency, and cost from the last 30 days.
  - When multiple providers are available, prefer the one with the best
    weighted score: quality * w_q + (1 - normalized_cost) * w_c + (1 - normalized_latency) * w_l
  - Falls back to rule-based routing when insufficient data exists.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .classifier import Classification
from .config import Config, ROOT_DIR, ensure_dirs
from .router import RouteDecision, Router
from .utils.logging import get_logger

_log = get_logger(__name__)


_ROUTE_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS route_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    task TEXT NOT NULL,
    domain TEXT NOT NULL,
    complexity REAL NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    quality_signal REAL NOT NULL DEFAULT 0.5,
    error TEXT DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_route_task ON route_outcomes(task, domain);
CREATE INDEX IF NOT EXISTS idx_route_provider ON route_outcomes(provider, model);
CREATE INDEX IF NOT EXISTS idx_route_ts ON route_outcomes(ts);
"""

ROUTE_HISTORY_DB = ROOT_DIR / "route_history.sqlite"


@dataclass
class RouteOutcome:
    """Recorded after each completion — feeds the learning loop."""

    provider: str
    model: str
    task: str
    domain: str
    complexity: float
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    quality_signal: float = 0.5
    error: Optional[str] = None


@dataclass
class ProviderPerformance:
    """Aggregated performance for a provider/model/task combination."""

    provider: str
    model: str
    avg_quality: float
    avg_latency_ms: float
    avg_cost: float
    sample_count: int
    error_rate: float


class AdaptiveRouter:
    """Routes based on historical outcomes when enough data exists.

    Falls back to the rule-based Router when data is insufficient.
    """

    # Minimum samples before we trust the history
    MIN_SAMPLES = 5
    # Lookback window in days
    LOOKBACK_DAYS = 30
    # Scoring weights
    DEFAULT_WEIGHTS = {"quality": 0.5, "cost": 0.3, "latency": 0.2}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._rule_router = Router(cfg)
        self._conn: Optional[sqlite3.Connection] = None
        self.weights = dict(self.DEFAULT_WEIGHTS)

    @property
    def providers(self):
        return self._rule_router.providers

    def _ensure_db(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        ensure_dirs()
        self._conn = sqlite3.connect(str(ROUTE_HISTORY_DB))
        self._conn.executescript(_ROUTE_HISTORY_SCHEMA)
        self._conn.commit()
        return self._conn

    def route(
        self,
        classification: Classification,
        force_local: bool = False,
        override_provider: Optional[str] = None,
        override_model: Optional[str] = None,
    ) -> RouteDecision:
        """Pick a provider+model using historical data when available."""
        # Overrides and forced modes bypass adaptive logic
        if override_provider or override_model or force_local:
            return self._rule_router.route(
                classification, force_local, override_provider, override_model
            )

        if self.cfg.mode == "local":
            return self._rule_router.route(classification, force_local=True)

        # Try adaptive routing
        decision = self._adaptive_route(classification)
        if decision is not None:
            return decision

        # Fall back to rule-based
        return self._rule_router.route(classification, force_local)

    def _adaptive_route(self, classification: Classification) -> Optional[RouteDecision]:
        """Score available providers by historical performance."""
        conn = self._ensure_db()
        cutoff = time.time() - (self.LOOKBACK_DAYS * 86400)
        task = classification.task

        # Get performance data for all providers on this task
        rows = conn.execute(
            """SELECT
                   provider, model,
                   AVG(quality_signal) as avg_quality,
                   AVG(latency_ms) as avg_latency,
                   AVG(cost_usd) as avg_cost,
                   COUNT(*) as sample_count,
                   SUM(CASE WHEN error IS NOT NULL THEN 1.0 ELSE 0.0 END) / COUNT(*) as error_rate
               FROM route_outcomes
               WHERE task = ? AND ts > ?
               GROUP BY provider, model
               HAVING COUNT(*) >= ?
            """,
            (task, cutoff, self.MIN_SAMPLES),
        ).fetchall()

        if not rows:
            return None

        # Score each candidate
        candidates: List[Tuple[float, str, str, ProviderPerformance]] = []
        max_cost = max(r[4] for r in rows) or 0.01
        max_latency = max(r[3] for r in rows) or 5000

        for r in rows:
            prov_name, model, avg_q, avg_l, avg_c, count, err_rate = r
            provider = self._rule_router.providers.get(prov_name)
            if not provider or not provider.is_available():
                continue
            if err_rate > 0.3:
                continue  # Skip unreliable providers

            perf = ProviderPerformance(
                provider=prov_name,
                model=model,
                avg_quality=avg_q,
                avg_latency_ms=avg_l,
                avg_cost=avg_c,
                sample_count=count,
                error_rate=err_rate,
            )

            # Weighted composite score
            quality_score = avg_q
            cost_score = 1.0 - (avg_c / max_cost) if max_cost > 0 else 1.0
            latency_score = 1.0 - (avg_l / max_latency) if max_latency > 0 else 1.0

            composite = (
                self.weights["quality"] * quality_score
                + self.weights["cost"] * cost_score
                + self.weights["latency"] * latency_score
            )

            candidates.append((composite, prov_name, model, perf))

        if not candidates:
            return None

        # Pick the best
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_prov, best_model, best_perf = candidates[0]

        provider = self._rule_router.providers[best_prov]
        fallbacks = [
            c[1]
            for c in candidates[1:3]
            if c[1] != best_prov
        ]

        return RouteDecision(
            provider=provider,
            model=best_model,
            reason=(
                f"adaptive: quality={best_perf.avg_quality:.2f} "
                f"cost=${best_perf.avg_cost:.4f} "
                f"latency={best_perf.avg_latency_ms:.0f}ms "
                f"(n={best_perf.sample_count})"
            ),
            fallback_providers=fallbacks or None,
            estimated_cost_usd=best_perf.avg_cost,
        )

    def record_outcome(self, outcome: RouteOutcome) -> None:
        """Record a completion result for future routing decisions."""
        try:
            conn = self._ensure_db()
            conn.execute(
                """INSERT INTO route_outcomes
                   (ts, provider, model, task, domain, complexity,
                    input_tokens, output_tokens, latency_ms, cost_usd,
                    quality_signal, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    outcome.provider,
                    outcome.model,
                    outcome.task,
                    outcome.domain,
                    outcome.complexity,
                    outcome.input_tokens,
                    outcome.output_tokens,
                    outcome.latency_ms,
                    outcome.cost_usd,
                    outcome.quality_signal,
                    outcome.error,
                ),
            )
            conn.commit()
        except Exception as e:
            _log.debug("failed to record outcome", extra={"error": str(e)})

    def complete_with_fallback(self, decision, messages, **kwargs):
        """Delegate to the rule router's fallback logic."""
        return self._rule_router.complete_with_fallback(decision, messages, **kwargs)

    def get_performance_report(
        self, lookback_days: int = 30, min_samples: int = 1
    ) -> List[ProviderPerformance]:
        """Get performance data for all provider/model combinations."""
        conn = self._ensure_db()
        cutoff = time.time() - (lookback_days * 86400)
        rows = conn.execute(
            """SELECT
                   provider, model,
                   AVG(quality_signal), AVG(latency_ms), AVG(cost_usd),
                   COUNT(*),
                   SUM(CASE WHEN error IS NOT NULL THEN 1.0 ELSE 0.0 END) / COUNT(*)
               FROM route_outcomes
               WHERE ts > ?
               GROUP BY provider, model
               HAVING COUNT(*) >= ?
               ORDER BY AVG(quality_signal) DESC
            """,
            (cutoff, min_samples),
        ).fetchall()
        return [
            ProviderPerformance(
                provider=r[0],
                model=r[1],
                avg_quality=r[2] or 0,
                avg_latency_ms=r[3] or 0,
                avg_cost=r[4] or 0,
                sample_count=r[5],
                error_rate=r[6] or 0,
            )
            for r in rows
        ]

    def reset_history(self) -> int:
        """Wipe all route history. Returns rows deleted."""
        conn = self._ensure_db()
        cur = conn.execute("DELETE FROM route_outcomes")
        conn.commit()
        return cur.rowcount
