"""Online Q-learning with linear function approximation for HRRL.

Implements the reinforcement learning component of the HRRL framework
(Keramati & Gutkin, 2014) adapted to epistemic agents.  The Q-learner
selects actions based on the agent's internal state (deficit, graph
density, stimulus count, message count) and updates its weights via
semi-gradient TD(0) using homeostatic reward (drive reduction).

Design decisions and justifications
------------------------------------
State features  φ(s) = [deficit, deficit², graph_density, n_stimuli, n_messages]
Actions         DEBATE_STIMULUS, DEBATE_PROACTIVE, SEARCH, READ, MESSAGE
Q-function      Q(s, a) = wₐᵀ φ(s)   (linear in features)
TD(0) update    wₐ ← wₐ + η · [r + γ · max_a' Q(s', a') − Q(s, a)] · φ(s)

Hyperparameters:
  η = 0.01   Learning rate.  Sutton & Barto (2018, §9.6) recommend
             η ≈ 1/(10·E[‖x‖²]) for linear FA.  With 5 features whose
             typical magnitude is O(1), E[‖x‖²] ≈ 5–10, giving
             η ≈ 1/50–1/100 ≈ 0.01–0.02.  We use η=0.01 (conservative).

  γ = 0.95   Discount factor.  Standard for continuing tasks with moderate
             horizon (Sutton & Barto 2018, §10.3).  The agent plans ~20
             steps ahead: 1/(1−γ) = 20, comparable to our 80-tick horizon.

  ε = 0.1    Exploration rate for ε-greedy policy (Sutton & Barto 2018,
             §2.3, §10.1).  With probability ε the agent selects a
             uniformly random action, ensuring all actions are tried and
             the Q-learner can discover which are genuinely useful.

Initialization:
  Weights are zero-initialized (Sutton & Barto 2018, §9.4).  This is
  the standard uninformative prior for linear FA — it assigns equal
  initial Q-value (0) to all actions, letting the reward signal alone
  determine the learned policy.  No action is favoured a priori.
"""

from __future__ import annotations

import numpy as np


ACTIONS = ("DEBATE_STIMULUS", "DEBATE_PROACTIVE", "SEARCH", "READ", "MESSAGE")
N_FEATURES = 5


def _build_features(
    deficit: float,
    graph_density: float,
    n_stimuli: int,
    n_messages: int = 0,
) -> np.ndarray:
    """Construct the state feature vector φ(s)."""
    return np.array([
        deficit,
        deficit ** 2,
        graph_density,
        float(n_stimuli),
        float(n_messages),
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
    ) -> None:
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
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
        self._weights[action] = (
            self._weights[action] + self.eta * td_error * features
        )

    def get_weights(self) -> dict[str, list[float]]:
        """Return current weight vectors for logging and analysis."""
        return {a: w.tolist() for a, w in self._weights.items()}

    @staticmethod
    def build_features(
        deficit: float,
        graph_density: float,
        n_stimuli: int,
        n_messages: int = 0,
    ) -> np.ndarray:
        """Public interface to construct state features."""
        return _build_features(deficit, graph_density, n_stimuli, n_messages)
