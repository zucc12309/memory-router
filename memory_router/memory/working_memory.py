"""Session-scoped working memory with automatic eviction.

Unlike the Memory Palace (long-term), working memory is a fixed-capacity
scratchpad for the current session. It evicts automatically when capacity is
exceeded, keeps relevance scores that decay per turn, and renders a compact
context block for the LLM.

Useful for: current file paths, error messages being debugged, variable names
in scope, project context for the current task, etc.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class WorkingSlot:
    """One item in working memory."""

    key: str
    value: Any
    relevance: float  # 0-1, decays within session
    created_turn: int
    last_accessed_turn: int


class WorkingMemory:
    """Fixed-capacity scratchpad for current session context.

    Slots are evicted LRU-by-relevance when capacity is exceeded. Relevance
    decays each turn so stale context is replaced by fresh context naturally.
    """

    def __init__(self, capacity: int = 20, token_budget: int = 2000):
        self._slots: OrderedDict[str, WorkingSlot] = OrderedDict()
        self._capacity = capacity
        self._token_budget = token_budget
        self._turn = 0

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def size(self) -> int:
        return len(self._slots)

    def advance_turn(self) -> None:
        """Advance the turn counter and decay all slot relevance."""
        self._turn += 1
        for slot in self._slots.values():
            age = self._turn - slot.last_accessed_turn
            slot.relevance *= max(0.5, 1.0 - (age * 0.1))

    def put(self, key: str, value: Any, relevance: float = 1.0) -> None:
        """Insert or update a working memory slot."""
        if key in self._slots:
            self._slots[key].value = value
            self._slots[key].relevance = min(1.0, relevance)
            self._slots[key].last_accessed_turn = self._turn
            self._slots.move_to_end(key)
            return
        while len(self._slots) >= self._capacity:
            min_key = min(self._slots, key=lambda k: self._slots[k].relevance)
            del self._slots[min_key]
        self._slots[key] = WorkingSlot(
            key=key,
            value=value,
            relevance=min(1.0, relevance),
            created_turn=self._turn,
            last_accessed_turn=self._turn,
        )

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value, boosting its relevance."""
        slot = self._slots.get(key)
        if slot is None:
            return None
        slot.last_accessed_turn = self._turn
        slot.relevance = min(1.0, slot.relevance + 0.1)
        return slot.value

    def remove(self, key: str) -> bool:
        """Remove a slot. Returns True if it existed."""
        if key in self._slots:
            del self._slots[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all slots."""
        self._slots.clear()

    def active_slots(self, min_relevance: float = 0.2) -> List[WorkingSlot]:
        """Return slots above the relevance threshold, sorted by relevance."""
        active = [s for s in self._slots.values() if s.relevance >= min_relevance]
        active.sort(key=lambda s: s.relevance, reverse=True)
        return active

    def snapshot_for_context(self, min_relevance: float = 0.2) -> str:
        """Render working memory as a context block for the LLM."""
        active = self.active_slots(min_relevance)
        if not active:
            return ""
        lines = [f"- {s.key}: {s.value}" for s in active]
        return "Current session context:\n" + "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for debugging/inspection."""
        return {
            "turn": self._turn,
            "capacity": self._capacity,
            "slots": [
                {
                    "key": s.key,
                    "value": s.value,
                    "relevance": round(s.relevance, 3),
                    "created_turn": s.created_turn,
                    "last_accessed_turn": s.last_accessed_turn,
                }
                for s in self._slots.values()
            ],
        }
