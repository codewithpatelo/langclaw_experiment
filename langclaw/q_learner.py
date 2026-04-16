"""Bounded online Q-learning adaptation for HRRL.

Keramati & Gutkin (2014) formulate HRRL over homeostatic reward and
state-action value learning. In this MAS we preserve that mathematical core:
reward is still drive reduction and action selection is still endogenous to
internal state. The extension is explicit: instead of a tabular state space,
we project discourse observables into a compact bounded feature vector and
learn a linear controller on top of that projection.

This file therefore implements an *adaptation* of HRRL, not a claim of
identity with the original tabular setting. The bounded feature map and the
clipping/regularisation below are included to keep long benchmark runs
numerically stable and scientifically interpretable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


ACTIONS = ("DEBATE_STIMULUS", "DEBATE_PROACTIVE", "SEARCH", "READ", "MESSAGE")
N_FEATURES = 5


def _build_features(
    deficit: float,
    graph_density: float,
    n_stimuli: int,
    n_messages: int = 0,
) -> np.ndarray:
    """Construct a bounded state feature vector φ(s).

    Mapping a continuous MAS state to bounded features is the main extension
    beyond the original HRRL setting. Every component is clipped to [0, 1] so
    the semi-gradient update remains well-conditioned over long runs.
    """
    deficit_hat = min(1.0, max(0.0, (deficit - 0.1) / 2.0))
    density_hat = min(1.0, max(0.0, graph_density))
    stimuli_hat = min(1.0, max(0.0, float(n_stimuli) / 5.0))
    messages_hat = min(1.0, max(0.0, float(n_messages) / 5.0))
    return np.array([
        deficit_hat,
        deficit_hat ** 2,
        density_hat,
        stimuli_hat,
        messages_hat,
    ], dtype=np.float64)


class HomeostaticQLearner:
    """Online Q-learner with linear function approximation and ε-greedy.

    Parameters
    ----------
    eta : float
        Learning rate for semi-gradient TD(0).  Justified by
        Sutton & Barto (2018, §9.6): η ≈ 1/(10·E[‖x‖²]).
    gamma : float
        Temporal discount factor.  γ=0.95 → effective horizon ≈ 20 steps.
    epsilon : float
        Exploration rate for ε-greedy action selection (Sutton & Barto
        2018, §2.3).  With probability ε, a uniformly random action is
        chosen.
    """

    def __init__(
        self,
        eta: float = 0.01,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        rng_seed: int | None = None,
        td_clip: float = 1.0,
        weight_clip: float = 5.0,
        l2_reg: float = 1e-4,
    ) -> None:
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.td_clip = td_clip
        self.weight_clip = weight_clip
        self.l2_reg = l2_reg
        self._rng = np.random.default_rng(rng_seed)

        self._weights: dict[str, np.ndarray] = {
            a: np.zeros(N_FEATURES, dtype=np.float64) for a in ACTIONS
        }

    def q_value(self, features: np.ndarray, action: str) -> float:
        """Compute Q(s, a) = wₐᵀ φ(s)."""
        return float(self._weights[action] @ features)

    def get_q_values(self, features: np.ndarray) -> dict[str, float]:
        """Return Q-values for all actions at the given state."""
        return {a: self.q_value(features, a) for a in ACTIONS}

    def select_action(self, features: np.ndarray) -> str:
        """ε-greedy action selection (Sutton & Barto 2018, §2.3).

        With probability ε: uniform random action (exploration).
        With probability 1−ε: argmax Q(s, a) (exploitation).
        """
        if self._rng.random() < self.epsilon:
            return self._rng.choice(ACTIONS)
        q_vals = self.get_q_values(features)
        return max(q_vals, key=q_vals.get)  # type: ignore[arg-type]

    def update(
        self,
        features: np.ndarray,
        action: str,
        reward: float,
        next_features: np.ndarray,
    ) -> None:
        """Semi-gradient TD(0) weight update.

        wₐ ← wₐ + η · [r + γ · max_a' Q(s', a') − Q(s, a)] · φ(s)
        """
        if action not in self._weights:
            return
        q_current = self.q_value(features, action)
        q_next_max = max(self.q_value(next_features, a) for a in ACTIONS)

        td_error = reward + self.gamma * q_next_max - q_current
        td_error = float(np.clip(td_error, -self.td_clip, self.td_clip))
        updated = self._weights[action] + self.eta * td_error * features
        updated *= (1.0 - self.eta * self.l2_reg)
        self._weights[action] = np.clip(updated, -self.weight_clip, self.weight_clip)

    def get_weights(self) -> dict[str, list[float]]:
        """Return current weight vectors for logging and analysis."""
        return {a: w.tolist() for a, w in self._weights.items()}

    def to_checkpoint(self) -> dict[str, Any]:
        """Serialize learner state for per-tick benchmark resume."""
        return {
            "eta": self.eta,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "td_clip": self.td_clip,
            "weight_clip": self.weight_clip,
            "l2_reg": self.l2_reg,
            "weights": self.get_weights(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_checkpoint(self, payload: dict[str, Any]) -> None:
        """Restore learner state from a serialized checkpoint."""
        self.eta = float(payload.get("eta", self.eta))
        self.gamma = float(payload.get("gamma", self.gamma))
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        self.td_clip = float(payload.get("td_clip", self.td_clip))
        self.weight_clip = float(payload.get("weight_clip", self.weight_clip))
        self.l2_reg = float(payload.get("l2_reg", self.l2_reg))

        for action, weights in payload.get("weights", {}).items():
            if action in self._weights:
                self._weights[action] = np.array(weights, dtype=np.float64)

        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state

    @staticmethod
    def build_features(
        deficit: float,
        graph_density: float,
        n_stimuli: int,
        n_messages: int = 0,
    ) -> np.ndarray:
        """Public interface to construct state features."""
        return _build_features(deficit, graph_density, n_stimuli, n_messages)
