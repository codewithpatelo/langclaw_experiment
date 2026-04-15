#!/usr/bin/env python3
"""
HRRL vs. Epistemic Energy Benchmark
=====================================
Compares the drive/activation functions of four published HRRL papers
against LangClaw's sigmoid epistemic energy function.

Use this to validate LangClaw's homeostatic mechanism against formal
mathematical baselines before publication.

References:
  [1] Keramati & Gutkin (2014) eLife 3:e04811 — HRRL Fundacional
  [2] Laurencon et al. (2024) arXiv:2401.08999 — CTCS-HRRL (tiempo continuo)
  [3] Wang et al. (ICLR 2025) arXiv:2412.06435 — D2A (deseos endógenos LLM)
  [4] Huang et al. (ICML 2025) arXiv:2408.00989 — MAS-Resilience (adversarial)

Usage:
  python hrrl_benchmark.py

To integrate your real epistemic energy function, replace the body of
`epistemic_energy()` with your sigmoid formula and adjust `E_THRESHOLD`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# §1  KERAMATI & GUTKIN 2014 — HRRL FUNDACIONAL
#     D(H_t) = Σi |h*_i - h_t,i|^(n/m)   [Eq. 1]
#     r(H_t, K_t) = D(H_t) - D(H_t + K_t)  [Eq. 2]
# ──────────────────────────────────────────────────────────────────────────────

def drive_keramati(H: np.ndarray, H_star: np.ndarray, m: float = 1.0, n: float = 1.0) -> float:
    """Homeostatic drive: generalised distance to set-point."""
    return float(np.sum(np.abs(H_star - H) ** (n / m)))


def reward_keramati(H: np.ndarray, K: np.ndarray, H_star: np.ndarray, m: float = 1.0, n: float = 1.0) -> float:
    """r = drive reduction after outcome K. Positive → homeostatic action."""
    return drive_keramati(H, H_star, m, n) - drive_keramati(H + K, H_star, m, n)


# ──────────────────────────────────────────────────────────────────────────────
# §2  CTCS-HRRL (Laurencon et al. 2024) — CONTINUOUS TIME-SPACE
#     d(δ) = √(ε + δᵀδ)           [Eq. 10]
#     r(t) = -δᵀδ̇ / √(ε + δᵀδ)   [Eq. 11]
#     Auto-decay: ζ̇ = f(ζ,u) ≈ cᵢ·(xᵢ + x*ᵢ)
#     c = [-0.05, -0.05, -0.008, +0.0005]
# ──────────────────────────────────────────────────────────────────────────────

CTCS_DECAY = np.array([-0.05, -0.05, -0.008, 0.0005])


def drive_ctcs(delta: np.ndarray, eps: float = 1e-8) -> float:
    """Euclidean norm of set-point deviation (numerically stable)."""
    return float(np.sqrt(eps + delta @ delta))


def reward_ctcs(delta: np.ndarray, delta_dot: np.ndarray, eps: float = 1e-8) -> float:
    """Instantaneous reward = -d(drive)/dt. Positive → convergence."""
    return float(-(delta @ delta_dot) / np.sqrt(eps + delta @ delta))


def ctcs_auto_decay(delta: np.ndarray, x_star: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Euler step of passive body dynamics (auto-regulation)."""
    x = delta + x_star
    delta_dot = CTCS_DECAY * x
    return delta + dt * delta_dot


# ──────────────────────────────────────────────────────────────────────────────
# §3  D2A — DESIRE-DRIVEN AUTONOMOUS AGENT (Wang et al. ICLR 2025)
#     π* = argmax_a Σd w_d · ΔV_d(a)   [Eq. 7]
#     V_{t+1} = V_t + α·(V_target - V_t)  [Eq. 8]
#     d_D2A = ||1 - V||                    [Eq. 9, implicit]
# ──────────────────────────────────────────────────────────────────────────────

DESIRE_DIMS = ["social", "fulfillment", "self_care", "safety",
               "autonomy", "knowledge", "economic", "belonging"]


@dataclass
class D2AAgent:
    desire_weights: np.ndarray = field(
        default_factory=lambda: np.ones(len(DESIRE_DIMS)) / len(DESIRE_DIMS)
    )
    desire_values: np.ndarray = field(
        default_factory=lambda: np.full(len(DESIRE_DIMS), 0.5)
    )
    alpha: float = 0.1

    def select_task(self, candidate_deltas: list[np.ndarray]) -> int:
        """π* = argmax_a Σd w_d · ΔV_d(a)"""
        scores = [float(self.desire_weights @ dv) for dv in candidate_deltas]
        return int(np.argmax(scores))

    def update_desires(self, v_target: np.ndarray) -> None:
        """V_{t+1} = V_t + α·(V_target - V_t)"""
        self.desire_values += self.alpha * (v_target - self.desire_values)

    def desire_drive(self) -> float:
        """Implicit drive: distance to full satisfaction."""
        return float(np.linalg.norm(1.0 - self.desire_values))


# ──────────────────────────────────────────────────────────────────────────────
# §4  MAS-RESILIENCE (Huang et al. ICML 2025) — ADVERSARIAL DEGRADATION
#     ΔP(e) ≈ 1 - exp(-k·e)
#     k_chain ≈ 0.301,  k_hier ≈ 0.057  (calibrated to paper results)
# ──────────────────────────────────────────────────────────────────────────────

K_CHAIN = -math.log(1 - 0.26)   # linear chain topology
K_HIER  = -math.log(1 - 0.055)  # hierarchical topology


def mas_degradation(error_rate: float, k: float) -> float:
    """Performance drop as a function of adversarial error rate (0→1)."""
    return 1.0 - math.exp(-k * error_rate)


# ──────────────────────────────────────────────────────────────────────────────
# §5  LANGCLAW — EPISTEMIC ENERGY (sigmoid + exponential decay)
#
#     E_ep(δ) = σ(k·(δ - θ))  [LangClaw activation, discretised from CTCS-HRRL]
#
#     Replace the body of epistemic_energy() with your actual formula
#     and adjust E_THRESHOLD to your model's set-point.
# ──────────────────────────────────────────────────────────────────────────────

E_THRESHOLD = 0.7   # θ in LangClaw


def epistemic_energy(deficit: float, k: float = 10.0, theta: float = 0.7) -> float:
    """
    LangClaw's sigmoid activation probability P(act | δ).

    This is the endogenous homeostatic trigger that replaces LangGraph's
    static conditional edges.  High deficit → high probability of acting.
    """
    return 1.0 / (1.0 + math.exp(-k * (deficit - theta)))


def homeostatic_trigger(deficit: float, theta: float = E_THRESHOLD) -> bool:
    """True when the epistemic drive exceeds the activation threshold."""
    return deficit >= theta


# ──────────────────────────────────────────────────────────────────────────────
# §6  COMPARISON HARNESS
# ──────────────────────────────────────────────────────────────────────────────

def run_comparison(T: int = 50, noise_sigma: float = 0.05) -> dict[str, list[float]]:
    rng = np.random.default_rng(42)

    x_star = np.array([1.0, 2.0])
    H = np.array([0.6, 1.5])
    delta_2d = H - x_star
    delta_4d = np.pad(delta_2d, (0, 2))
    x_star_4d = np.pad(x_star, (0, 2))

    agent = D2AAgent()
    deficit = 0.1  # LangClaw initial deficit

    results: dict[str, list[float]] = {k: [] for k in [
        "HRRL_Keramati", "CTCS_HRRL", "D2A_Drive",
        "MAS_Chain", "MAS_Hier", "LangClaw_EpistemicEnergy"
    ]}

    for t in range(T + 1):
        results["HRRL_Keramati"].append(drive_keramati(H, x_star))
        results["CTCS_HRRL"].append(drive_ctcs(delta_4d))
        results["D2A_Drive"].append(agent.desire_drive())
        results["MAS_Chain"].append(mas_degradation(t / T, K_CHAIN))
        results["MAS_Hier"].append(mas_degradation(t / T, K_HIER))
        results["LangClaw_EpistemicEnergy"].append(epistemic_energy(deficit))

        if t < T:
            # Adversarial drift
            adv = rng.normal(0, noise_sigma, 2) - 0.02
            H = H + adv
            delta_2d = H - x_star
            delta_4d = ctcs_auto_decay(np.pad(delta_2d, (0, 2)), x_star_4d, dt=0.05)
            agent.update_desires(rng.uniform(0.4, 0.9, len(DESIRE_DIMS)))

            # LangClaw deficit dynamics (decay + partial satiation)
            deficit += 0.05              # linear decay
            if epistemic_energy(deficit) > 0.5:
                deficit = max(0.1, deficit - 0.5 * 0.3)  # simulated satiation

    return results


def print_table(results: dict[str, list[float]], T: int = 50) -> None:
    print(f"\n{'Modelo':<30} {'t=0':>8} {'t=25':>8} {'t=50':>8} {'D(50-0)':>12}")
    print("-" * 62)
    for name, vals in results.items():
        mid = vals[T // 2]
        delta = vals[-1] - vals[0]
        print(f"{name:<30} {vals[0]:>8.4f} {mid:>8.4f} {vals[-1]:>8.4f} {delta:>+12.4f}")

    trigger_at_25 = homeostatic_trigger(
        deficit=0.1 + 0.05 * 25  # rough deficit at t=25 without satiation
    )
    print(f"\nLangClaw trigger at t=25 (no satiation): {trigger_at_25}")
    print(f"Episodic memory gap vs D2A: D2A has open update loop, no formal stopping criterion.")
    print(f"LangClaw provides the closed homeostatic loop D2A is missing.")


if __name__ == "__main__":
    results = run_comparison(T=50)
    print_table(results)
