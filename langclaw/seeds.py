"""Deterministic prime seed factory for reproducible experiments.

Given a single integer ``master_seed``, ``SeedFactory`` derives one unique
prime seed per named component (agent RNG, agent LLM, simulation RNG, etc.).

Design principles
-----------------
- **Deterministic**: uses SHA-256 of ``(master_seed, component_name)`` so
  the seed for each component is stable regardless of call order.
- **Prime output**: primes have better spacing than consecutive integers and
  reduce accidental correlations between independent PRNGs.
- **No PYTHONHASHSEED dependency**: Python's built-in ``hash()`` is randomised
  by default; SHA-256 is always stable across processes and platforms.

Usage
-----
    factory = SeedFactory(42)
    factory.get("simulation")          # e.g. 104723
    factory.get("agent_GOV-1_rng")     # e.g. 98317
    factory.get("agent_GOV-1_llm")     # e.g. 112337
    factory.summary()                  # dict of all derived seeds so far
"""

from __future__ import annotations

import hashlib
from typing import Iterator


# ──────────────────────────────────────────────────────────────────────────────
# Prime utilities
# ──────────────────────────────────────────────────────────────────────────────

def is_prime(n: int) -> bool:
    """Miller-Rabin-free deterministic primality test (sufficient for n<2^31)."""
    if n < 2:
        return False
    if n in (2, 3, 5, 7):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def next_prime(n: int) -> int:
    """Return the smallest prime >= n."""
    if n <= 2:
        return 2
    candidate = n | 1  # ensure odd
    while not is_prime(candidate):
        candidate += 2
    return candidate


# ──────────────────────────────────────────────────────────────────────────────
# SeedFactory
# ──────────────────────────────────────────────────────────────────────────────

# Search range for prime candidates: [PRIME_MIN, PRIME_MIN + PRIME_RANGE)
# Large enough to avoid collisions for any realistic number of components.
_PRIME_MIN: int = 10_007      # first prime > 10_000
_PRIME_RANGE: int = 900_000   # search window (~74k primes in this range)


class SeedFactory:
    """Derives deterministic prime seeds from a single master seed.

    Parameters
    ----------
    master_seed : int
        The experiment-level seed. Pass via ``--seed`` on the CLI.

    Notes
    -----
    Collision probability: with ~74k primes in the search window and a uniform
    SHA-256 distribution, two components sharing a seed has probability
    1/74000 ≈ 1.4e-5.  Acceptable for up to ~100 named components.
    """

    def __init__(self, master_seed: int) -> None:
        self.master_seed = master_seed
        self._cache: dict[str, int] = {}

    def get(self, component: str) -> int:
        """Return the prime seed for *component*, derived from master_seed.

        Repeated calls with the same component always return the same value.
        """
        if component not in self._cache:
            self._cache[component] = self._derive(component)
        return self._cache[component]

    def summary(self) -> dict[str, int]:
        """Return all seeds derived so far (for logging / paper appendix)."""
        return dict(self._cache)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _derive(self, component: str) -> int:
        raw = self._sha256_int(self.master_seed, component)
        # Map into [PRIME_MIN, PRIME_MIN + PRIME_RANGE) and find next prime
        offset = raw % _PRIME_RANGE
        return next_prime(_PRIME_MIN + offset)

    @staticmethod
    def _sha256_int(master_seed: int, component: str) -> int:
        data = f"{master_seed}:{component}".encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        return int(digest, 16)
