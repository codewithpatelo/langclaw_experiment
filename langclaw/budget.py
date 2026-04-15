"""API Budget manager for LangClaw.

Two-tier rate limiting per agent:
  - Hard limit: absolute ceiling on total API calls per run (functional
    constraint — set by real API cost ceiling).
  - Soft limit: sliding-window cap derived from the sigmoid activation
    function, ensuring budget allocation is proportional to the agent's
    activation probability.

Soft-cap derivation
-------------------
The sigmoid gate p = σ(k·(δ − θ)) maps deficit to [0, 1].  We use this
same function to determine how many API calls the agent may make per
window, discretised into {1, 2, 3, 4}:

    soft_cap(δ) = max(1, round(p(δ) · MAX_CALLS_PER_WINDOW))

This ensures:
  - The budget thresholds are derived from the same sigmoid parameters
    (k=10, θ=0.7) used in the activation gate — no independent arbitrary
    thresholds.
  - A sated agent (p ≈ 0) gets 1 call/window (minimum to stay responsive).
  - A maximally driven agent (p ≈ 1) gets MAX_CALLS_PER_WINDOW calls.
  - The transition is smooth, not stepped.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque


MAX_CALLS_PER_WINDOW: int = 4


class APIBudget:
    """Per-agent API call limiter with sigmoid-derived soft cap.

    Parameters
    ----------
    hard_limit : int
        Maximum total API calls per agent for the entire run.
        Functional constraint: set to match real API cost ceiling.
    window_size : int
        Sliding window in ticks.  With 80 total ticks and 10 agents,
        window=5 allows ~16 windows, giving enough granularity for
        budget to respond to deficit changes.
    k : float
        Sigmoid steepness — must match homeostasis.py (default 10.0).
    theta : float
        Sigmoid midpoint — must match homeostasis.py (default 0.7).
    """

    def __init__(
        self,
        hard_limit: int = 200,
        window_size: int = 5,
        k: float = 10.0,
        theta: float = 0.7,
    ) -> None:
        self.hard_limit = hard_limit
        self.window_size = window_size
        self._k = k
        self._theta = theta
        self._total_calls: dict[str, int] = defaultdict(int)
        self._window_calls: dict[str, deque[int]] = defaultdict(deque)

    def can_call(self, agent_id: str, deficit: float, current_tick: int) -> bool:
        """Return True if the agent is allowed to make an API call."""
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

    def _evict_old(self, agent_id: str, current_tick: int) -> None:
        """Remove calls that have fallen outside the sliding window."""
        q = self._window_calls[agent_id]
        while q and current_tick - q[0] >= self.window_size:
            q.popleft()

    def _soft_cap(self, deficit: float) -> int:
        """Sigmoid-derived soft cap: same function as the activation gate.

        soft_cap = max(1, round(σ(k·(δ−θ)) · MAX_CALLS_PER_WINDOW))
        """
        exponent = -self._k * (deficit - self._theta)
        exponent = max(-500.0, min(500.0, exponent))
        p = 1.0 / (1.0 + math.exp(exponent))
        return max(1, round(p * MAX_CALLS_PER_WINDOW))
