"""Homeostatic Regulation via Reinforcement Learning (HRRL) module.

Implements the epistemic drive that regulates agent participation through
an internal deficit signal, analogous to biological homeostasis:

- The deficit grows over time via ``decay`` (hunger signal).
- A sigmoid activation function maps deficit to action probability.
- Successful, high-quality contributions reduce the deficit via ``satiate``.

Mathematical foundation
-----------------------
Activation probability (sigmoid):
    p = 1 / (1 + exp(-k * (deficit - θ)))

Satiation update:
    deficit_new = max(ε, deficit - α * Δφ*)

where ε = 0.1 is the resting baseline, α is the learning rate,
and Δφ* is the informational quality proxy of the contribution.
"""

from __future__ import annotations

import math


class EpistemicDrive:
    """Internal homeostatic regulator for a single agent.

    Parameters
    ----------
    initial_deficit : float
        Starting deficit level (default 0.1, the resting baseline).
    """

    BASELINE: float = 0.1

    def __init__(self, initial_deficit: float = 0.1) -> None:
        self.deficit: float = max(self.BASELINE, initial_deficit)
        self._history: list[float] = [self.deficit]

    def decay(self, lambda_rate: float = 0.05) -> None:
        """Increase deficit by a fixed rate each tick (drive accumulation).

        Models the passage of time without epistemic contribution:
            deficit += λ
        """
        self.deficit += lambda_rate
        self._history.append(self.deficit)

    def get_activation_probability(self, k: float = 10.0, theta: float = 0.7) -> float:
        """Return probability of acting via a sigmoid function.

        p = 1 / (1 + exp(-k * (deficit - θ)))

        When deficit < θ the agent is unlikely to act;
        when deficit > θ the agent is compelled to participate.
        """
        exponent = -k * (self.deficit - theta)
        exponent = max(-500.0, min(500.0, exponent))
        return 1.0 / (1.0 + math.exp(exponent))

    def satiate(self, delta_phi: float, alpha: float = 0.5) -> None:
        """Reduce deficit proportionally to contribution quality.

        deficit = max(ε, deficit - α * Δφ*)

        If Δφ* = 0 (isolated monologue with no informational value),
        the deficit does not decrease, penalising low-quality output.
        """
        self.deficit = max(self.BASELINE, self.deficit - alpha * delta_phi)
        self._history.append(self.deficit)

    @property
    def history(self) -> list[float]:
        """Full deficit trajectory for plotting."""
        return list(self._history)
