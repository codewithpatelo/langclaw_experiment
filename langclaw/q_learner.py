"""Online Q-learning with linear function approximation for HRRL.

Implements the reinforcement learning component of the HRRL framework
(Keramati & Gutkin, 2014) adapted to epistemic agents.  The Q-learner
selects actions based on the agent's internal state (deficit, graph
density, stimulus count) and updates its weights via TD(0) using
homeostatic reward (drive reduction).

State features  phi(s) = [deficit, deficit^2, graph_density, n_stimuli]
Actions         DEBATE_STIMULUS, DEBATE_PROACTIVE, SEARCH, READ
Q-function      Q(s, a) = w_a^T phi(s)   (linear in features)
TD(0) update    w_a <- w_a + eta * [r + gamma * max_a' Q(s', a') - Q(s, a)] * phi(s)
"""

from __future__ import annotations

import numpy as np


ACTIONS = ("DEBATE_STIMULUS", "DEBATE_PROACTIVE", "SEARCH", "READ")
N_FEATURES = 4


def _build_features(
    deficit: float,
    graph_density: float,
    n_stimuli: int,
) -> np.ndarray:
    """Construct the state feature vector phi(s)."""
    return np.array([
        deficit,
        deficit ** 2,
        graph_density,
        float(n_stimuli),
    ], dtype=np.float64)


class HomeostaticQLearner:
    """Online Q-learner with linear function approximation.

    Parameters
    ----------
    eta : float
        Learning rate for TD(0) updates.
    gamma : float
        Temporal discount factor.  Keramati shows that discounting
        motivates shortest-path behavior toward the homeostatic setpoint.
    """

    def __init__(self, eta: float = 0.01, gamma: float = 0.95) -> None:
        self.eta = eta
        self.gamma = gamma

        self._weights: dict[str, np.ndarray] = {
            a: np.zeros(N_FEATURES, dtype=np.float64) for a in ACTIONS
        }
        self._warm_start()

    def _warm_start(self) -> None:
        """Initialize weights so the Q-function produces sensible values
        before any learning has occurred.

        The heuristic: DEBATE actions should have moderate positive weight
        on deficit (higher deficit -> more value in debating), and slight
        negative weight on graph density (less value when graph is already
        dense for proactive debate, more for stimulus debate).
        SEARCH/READ have lower initial values.
        """
        self._weights["DEBATE_STIMULUS"] = np.array(
            [0.4, 0.1, -0.05, 0.2], dtype=np.float64
        )
        self._weights["DEBATE_PROACTIVE"] = np.array(
            [0.3, 0.1, -0.15, -0.05], dtype=np.float64
        )
        self._weights["SEARCH"] = np.array(
            [0.1, 0.0, -0.1, -0.1], dtype=np.float64
        )
        self._weights["READ"] = np.array(
            [0.05, 0.0, -0.05, -0.05], dtype=np.float64
        )

    def q_value(self, features: np.ndarray, action: str) -> float:
        """Compute Q(s, a) = w_a^T phi(s)."""
        return float(self._weights[action] @ features)

    def get_q_values(self, features: np.ndarray) -> dict[str, float]:
        """Return Q-values for all actions at the given state."""
        return {a: self.q_value(features, a) for a in ACTIONS}

    def select_action(self, features: np.ndarray) -> str:
        """Return the action with highest Q-value (greedy policy)."""
        q_vals = self.get_q_values(features)
        return max(q_vals, key=q_vals.get)  # type: ignore[arg-type]

    def update(
        self,
        features: np.ndarray,
        action: str,
        reward: float,
        next_features: np.ndarray,
    ) -> None:
        """TD(0) weight update.

        w_a <- w_a + eta * [r + gamma * max_a' Q(s', a') - Q(s, a)] * phi(s)
        """
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
    ) -> np.ndarray:
        """Public interface to construct state features."""
        return _build_features(deficit, graph_density, n_stimuli)
