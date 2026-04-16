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

Hyperparameter justification
----------------------------
All parameters form a coherent dynamical system.  The design goal is:
an agent that starts neutral should reach 50% activation probability
around tick 8–12 (ramp-up phase), and a high-quality action (Δφ* ≈ 0.15)
should return the deficit to near-setpoint levels.

  ε = 0.1    Homeostatic setpoint.  Derived from the drive function's
             boundary condition: D(ε) = (ε − ε)^m = 0, so any deficit
             at setpoint produces zero drive.  The value 0.1 (not 0)
             prevents numerical instability and ensures satiation can
             never fully extinguish the deficit (Keramati & Gutkin 2014,
             eq. 2).

  λ = 0.05   Basal decay rate (deficit += λ per tick).  Chosen so that
             θ is reached in (θ − δ₀)/λ ticks from initial deficit δ₀.
             With δ₀ = 0.5 and θ = 0.7: (0.7 − 0.5)/0.05 = 4 ticks
             to 50% activation.  This produces a brief warm-up before
             full engagement, consistent with biological ramp-up
             (Keramati & Gutkin 2014, §3.2).

  θ = 0.7    Sigmoid midpoint (50% activation threshold).  Derived from
             the satiation balance: after a good debate action with
             Δφ* ≈ 0.15, deficit reduces by α·Δφ* = 2.0·0.15 = 0.30.
             Setting θ = 0.7 means the agent needs deficit ≈ 0.7 to be
             likely to act, and a single good action (reducing by 0.30)
             brings it to ≈ 0.4, below threshold — creating the
             oscillatory act-rest cycle that characterises homeostasis.

  k = 10     Sigmoid steepness.  Controls the transition sharpness.
             At k=10: p(δ=θ−0.1) ≈ 0.27, p(δ=θ) = 0.50,
             p(δ=θ+0.1) ≈ 0.73.  A range of ±0.1 around θ spans
             [0.27, 0.73], giving a smooth but decisive transition.
             Lower k (e.g., 5) makes the gate too fuzzy; higher k
             (e.g., 20) makes it almost binary.  k=10 is the standard
             choice for logistic activation with unit-scale inputs
             (Bishop 2006, §4.2).

  γ_stim = 0.1  Stimulus coupling coefficient.  A maximally relevant
             stimulus (relevance=1.0) increases deficit by 0.1, equivalent
             to 2 ticks of basal decay.  This ensures that a single
             important event can push a near-threshold agent into action
             without overwhelming the homeostatic dynamics.

  α = 2.0    Satiation gain for DEBATE (the only action with measured
             Δφ*).  With typical Δφ* ∈ [0.05, 0.20], satiation
             α·Δφ* ∈ [0.10, 0.40], enough to cross below θ after a
             good contribution.  Other actions compute their own
             outcome-dependent satiation factor (see agent.py).
"""

from __future__ import annotations

import math


class EpistemicDrive:
    """Internal homeostatic regulator for a single agent.

    Parameters
    ----------
    initial_deficit : float
        Starting deficit level.  Default 0.5 places the agent below θ
        (p(act) ≈ 12%), requiring ~4 ticks of decay to reach 50%.
    m : int
        Drive function exponent.  m=2 (Keramati & Gutkin 2014, quadratic
        case): superlinear pressure from large deviations.
    """

    BASELINE: float = 0.1

    def __init__(self, initial_deficit: float = 0.5, m: int = 2) -> None:
        self.deficit: float = max(self.BASELINE, initial_deficit)
        self.m: int = m
        self._history: list[float] = [self.deficit]

    def decay(self, lambda_rate: float = 0.05) -> None:
        """Increase deficit by λ per tick (basal drive accumulation).

        Justified: λ=0.05 gives (θ−δ₀)/λ = (0.7−0.5)/0.05 = 4 ticks
        to reach 50% activation from default initial deficit.
        """
        self.deficit += lambda_rate
        self._history.append(self.deficit)

    def get_activation_probability(self, k: float = 10.0, theta: float = 0.7) -> float:
        """Sigmoid activation gate: p = σ(k·(δ − θ)).

        k=10, θ=0.7: transition band [0.27, 0.73] spans δ ∈ [0.6, 0.8].
        See module docstring for full derivation.
        """
        exponent = -k * (self.deficit - theta)
        exponent = max(-500.0, min(500.0, exponent))
        return 1.0 / (1.0 + math.exp(exponent))

    def stimulate(self, relevance: float, gamma: float = 0.1) -> None:
        """Increase deficit in response to a relevant stimulus.

        deficit += γ · relevance.  γ=0.1: a maximally relevant event
        adds 0.1 (≡ 2 ticks of decay), making stimulus-driven
        activation comparable to but not dominant over basal accumulation.
        """
        self.deficit += gamma * relevance
        self._history.append(self.deficit)

    def satiate(self, delta_phi: float, alpha: float = 2.0) -> None:
        """Reduce deficit proportionally to contribution quality.

        deficit = max(ε, δ − α·Δφ*).  α=2.0 for DEBATE ensures that
        a typical Δφ*=0.15 reduces deficit by 0.30, crossing below θ.
        Other actions pass their own outcome-dependent delta_phi.
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

    def to_checkpoint(self) -> dict[str, object]:
        """Serialize internal homeostatic state."""
        return {
            "deficit": self.deficit,
            "m": self.m,
            "history": list(self._history),
        }

    def load_checkpoint(self, payload: dict[str, object]) -> None:
        """Restore internal homeostatic state."""
        self.deficit = float(payload.get("deficit", self.deficit))
        self.m = int(payload.get("m", self.m))
        history = payload.get("history")
        if isinstance(history, list) and history:
            self._history = [float(x) for x in history]
        else:
            self._history = [self.deficit]
