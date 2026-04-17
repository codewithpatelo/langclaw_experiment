"""AHP (Analytic Hierarchy Process) weight derivation for the LLM-as-judge
rubric used in the LangClaw post-hoc evaluation.

Six criteria (per the deep-research design):

    C1 Relevancia tematica
    C2 Novedad argumentativa
    C3 Calidad logica
    C4 Especificidad del ataque
    C5 Ausencia de falacias
    C6 Coherencia con contexto

The pairwise matrix is filled with Saaty 1-9 scale values reflecting the
following reasoned priorities for adversarial debate where context
collapse is the central phenomenon under study:

  * C6 (coherencia con contexto) is the most important: it is the very
    failure mode the experiment targets.
  * C3 (calidad logica) and C4 (especificidad del ataque) are next:
    structural coherence depends on these.
  * C1 (relevancia tematica) is foundational but coarser-grained.
  * C5 (ausencia de falacias) overlaps partially with C3.
  * C2 (novedad argumentativa) is secondary in adversarial debate where
    well-targeted repetition can be useful.

We compute weights via the principal eigenvector method, and the
Consistency Ratio (CR) using Saaty's Random Index for n=6 (RI=1.24).
A CR below 0.10 is required for the matrix to be considered consistent.
"""
from __future__ import annotations

import numpy as np


CRITERIA = ["C1 Relevancia", "C2 Novedad", "C3 CalidadLog",
            "C4 EspecAtaque", "C5 Falacias", "C6 Coherencia"]

RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
            7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


# Saaty pairwise matrix. Reciprocity enforced below.
# Row i, column j: how much C_i dominates C_j (1=equal, 9=extreme).
A = np.array([
    # C1   C2    C3    C4    C5   C6
    [1.0, 3.0, 1/2, 1/2, 2.0, 1/3],   # C1 vs others
    [1/3, 1.0, 1/3, 1/3, 1/2, 1/4],   # C2
    [2.0, 3.0, 1.0, 1.0, 2.0, 1/2],   # C3
    [2.0, 3.0, 1.0, 1.0, 2.0, 1/2],   # C4
    [1/2, 2.0, 1/2, 1/2, 1.0, 1/3],   # C5
    [3.0, 4.0, 2.0, 2.0, 3.0, 1.0],   # C6
])


def enforce_reciprocity(M: np.ndarray) -> np.ndarray:
    """Saaty matrices must satisfy a_ji = 1 / a_ij. Average upper and
    lower triangles to absorb any rounding inconsistency, then enforce
    reciprocity strictly from the upper triangle."""
    n = M.shape[0]
    out = np.copy(M)
    for i in range(n):
        for j in range(i + 1, n):
            out[j, i] = 1.0 / out[i, j]
        out[i, i] = 1.0
    return out


def ahp_weights(M: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Compute AHP weights via principal eigenvector and the Consistency
    Ratio (CR). Returns (weights, lambda_max, CR)."""
    n = M.shape[0]
    eigvals, eigvecs = np.linalg.eig(M)
    idx = int(np.argmax(eigvals.real))
    lam_max = float(eigvals[idx].real)
    w = np.abs(eigvecs[:, idx].real)
    w = w / w.sum()
    ci = (lam_max - n) / (n - 1)
    ri = RI_TABLE[n]
    cr = ci / ri
    return w, lam_max, cr


def main() -> None:
    M = enforce_reciprocity(A)
    w, lam_max, cr = ahp_weights(M)

    print("=" * 64)
    print("AHP weights for LangClaw LLM-as-judge rubric (n=6 criteria)")
    print("=" * 64)
    print()
    print("Pairwise comparison matrix (Saaty 1-9):")
    header = "        " + "  ".join(f"{c[:5]:>6s}" for c in CRITERIA)
    print(header)
    for i, name in enumerate(CRITERIA):
        row = "  ".join(f"{M[i, j]:6.3f}" for j in range(M.shape[1]))
        print(f"{name[:5]:>6s}  {row}")

    print()
    print("Resulting weights:")
    for name, weight in zip(CRITERIA, w):
        bar = "#" * int(round(weight * 60))
        print(f"  {name:<18s}  w = {weight:.4f}  {bar}")
    print(f"\n  sum(w) = {w.sum():.6f}")

    print()
    print(f"  lambda_max     = {lam_max:.4f}")
    print(f"  CI             = {(lam_max - 6) / 5:.4f}")
    print(f"  RI (n=6)       = {RI_TABLE[6]:.4f}")
    print(f"  CR             = {cr:.4f}  -> "
          f"{'CONSISTENT (<0.10)' if cr < 0.10 else 'INCONSISTENT (>=0.10)'}")

    print()
    print("Reportable LaTeX-friendly summary:")
    print("  weights = [" + ", ".join(f"{x:.4f}" for x in w) + "]")
    print(f"  CR      = {cr:.4f}")


if __name__ == "__main__":
    main()
