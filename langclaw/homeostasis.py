"""Homeostatic Regulation via Reinforcement Learning (HRRL) module.

Faithful implementation of the HRRL framework (Keramati & Gutkin, 2014)
adapted to epistemic agents.  The core components:

1. **Drive function** D(δ) = (δ - ε)^m  -- maps deficit to motivational
   intensity.  Quadratic (m=2) by default: large deviations from setpoint
   produce superlinear pressure, consistent with Keramati's nonlinear
   mapping between homeostatic deviation and motivation.

2. **Reward as drive reduction** r_t = D(δ_t) - D(δ_{t+1})  -- the central
   axiom of HRRL.  Reward is not an external signal; it is defined as the
   decrease in drive caused by the agent's action.  By this definition,
   the equivalence theorem (Keramati & Gutkin, 2014, Theorem 1) guarantees
   that maximizing expected discounted reward simultaneously minimizes
   expected future drive.

3. **State transitions** -- deficit grows via basal decay (λ), increases
   via discourse stimuli (γ·r), and decreases via quality-proportional
   satiation (α·Δφ*).  These govern how the internal state evolves;
   reward is derived from the resulting drive change.

4. **Sigmoid activation** p = σ(k·(δ - θ))  -- maps deficit to action
   probability, gating execution via Bernoulli sampling.
"""

from __future__ import annotations

import math


class EpistemicDrive:
    """Internal homeostatic regulator for a single agent.

    Parameters
    ----------
    initial_deficit : float
        Starting deficit level (default 0.5).
    m : int
        Drive function exponent (default 2, Keramati's quadratic case).
    """

    BASELINE: float = 0.1  # ε -- homeostatic setpoint

    def __init__(self, initial_deficit: float = 0.1, m: int = 2) -> None:
        self.deficit: float = max(self.BASELINE, initial_deficit)
        self.m: int = m
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

    def stimulate(self, relevance: float, gamma: float = 0.1) -> None:
        """Increase deficit in response to a relevant stimulus.

        deficit += gamma * relevance

        A relevant event (e.g. an attack on the agent's faction with high
        centrality) pushes the deficit up, making the agent more likely to
        act.  This replaces the blind-timer accumulation model: the deficit
        now integrates both basal decay and discourse-driven stimuli.
        """
        self.deficit += gamma * relevance
        self._history.append(self.deficit)

    def satiate(self, delta_phi: float, alpha: float = 0.5) -> None:
        """Reduce deficit proportionally to contribution quality.

        deficit = max(ε, deficit - α * Δφ*)

        If Δφ* = 0 (isolated monologue with no informational value),
        the deficit does not decrease, penalising low-quality output.
        """
        self.deficit = max(self.BASELINE, self.deficit - alpha * delta_phi)
        self._history.append(self.deficit)

    @property
    def drive_value(self) -> float:
        """Drive function D(δ) = (δ - ε)^m.

        Quadratic by default (m=2): large deviations from setpoint produce
        superlinear motivational pressure.
        """
        return (self.deficit - self.BASELINE) ** self.m

    @staticmethod
    def compute_reward(
        delta_before: float,
        delta_after: float,
        epsilon: float = 0.1,
        m: int = 2,
    ) -> float:
        """Homeostatic reward = drive reduction caused by the action.

        r = D(δ_before) - D(δ_after)

        Positive reward when the action reduces drive (deficit decreases);
        negative reward when drive increases (e.g., decay without action).
        """
        d_before = (delta_before - epsilon) ** m
        d_after = (delta_after - epsilon) ** m
        return d_before - d_after

    @property
    def history(self) -> list[float]:
        """Full deficit trajectory for plotting."""
        return list(self._history)
