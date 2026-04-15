"""
API Budget manager for LangClaw.

Two-tier rate limiting per agent:
  - Hard limit: absolute ceiling on total API calls per run (budget protection)
  - Soft limit: sliding-window cap that scales with the agent's epistemic deficit
    (sated agents call less; hungry agents can call more within the hard ceiling)
"""

from __future__ import annotations
from collections import defaultdict, deque


class APIBudget:
    """
    Per-agent API call limiter.

    The soft cap is a function of the agent's current deficit δ:
      δ < 0.3  → 1 call per window  (agent is sated, hold back)
      δ < 0.7  → 2 calls per window (moderate drive)
      δ >= 0.7 → 4 calls per window (high drive, agent is hungry)

    The hard limit is a global ceiling regardless of δ.
    """

    def __init__(self, hard_limit: int = 200, window_size: int = 5) -> None:
        """
        Args:
            hard_limit:  Maximum total API calls per agent for the entire run.
            window_size: Number of ticks in the sliding window for soft-cap tracking.
        """
        self.hard_limit = hard_limit
        self.window_size = window_size
        self._total_calls: dict[str, int] = defaultdict(int)
        # Stores the tick number of each call in the window
        self._window_calls: dict[str, deque[int]] = defaultdict(deque)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_call(self, agent_id: str, deficit: float, current_tick: int) -> bool:
        """Return True if the agent is allowed to make an API call right now."""
        if self._total_calls[agent_id] >= self.hard_limit:
            return False
        self._evict_old(agent_id, current_tick)
        soft_cap = self._soft_cap(deficit)
        return len(self._window_calls[agent_id]) < soft_cap

    def record_call(self, agent_id: str, current_tick: int) -> None:
        """Register that agent_id made an API call at current_tick."""
        self._total_calls[agent_id] += 1
        self._window_calls[agent_id].append(current_tick)

    def total_calls(self, agent_id: str) -> int:
        return self._total_calls[agent_id]

    def remaining(self, agent_id: str) -> int:
        return max(0, self.hard_limit - self._total_calls[agent_id])

    def summary(self) -> dict[str, int]:
        return dict(self._total_calls)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_old(self, agent_id: str, current_tick: int) -> None:
        """Remove calls that have fallen outside the sliding window."""
        q = self._window_calls[agent_id]
        while q and current_tick - q[0] >= self.window_size:
            q.popleft()

    def _soft_cap(self, deficit: float) -> int:
        if deficit < 0.3:
            return 1
        if deficit < 0.7:
            return 2
        return 4
