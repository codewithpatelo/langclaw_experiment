"""Re-analyse inter-rater agreement between offline LLM judges.

Motivation
----------
The original agreement statistic (unweighted Cohen's kappa over tercile bins)
is inappropriate for the data at hand:

  1. Binning 1-10 scores into three categories discards ordinal information.
     Unweighted kappa then treats a 7-vs-8 disagreement identically to a
     2-vs-9 disagreement -- no partial credit for near misses.
  2. Both judges concentrate in a narrow band (7-8), inflating chance
     agreement p_e and depressing kappa = (p_0 - p_e) / (1 - p_e).
  3. A systematic calibration offset between judges (DeepSeek scores higher
     than GLM on average) shifts identical arguments into different bins even
     when the judges rank them identically.

This script recomputes agreement on the raw scores using statistics that are
appropriate for ordinal / continuous ratings:

  - Cohen's kappa, quadratic weights: standard for ordinal scales; penalises
    disagreement in proportion to squared distance.
  - Spearman rho: rank correlation; invariant to monotone recalibration, so it
    isolates whether judges *order* arguments consistently.
  - Pearson r: linear correlation on raw scores.
  - ICC(2,1): two-way random effects, single rater, absolute agreement.
    The conventional reliability coefficient for continuous ratings.
  - Calibration offset: mean signed difference between judges, plus the
    Bland-Altman limits of agreement.

Inputs are read from the judge checkpoint written by run_judge_smoke.py, so no
API calls are made.

Usage
-----
    python analyze_judge_agreement.py
    python analyze_judge_agreement.py --checkpoint judge_smoke_checkpoint.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = BASE_DIR / "judge_smoke_checkpoint.json"
DEFAULT_OUTPUT = BASE_DIR / "judge_agreement_results.json"

MODES = ["epr", "epr_q", "epr_sham", "langgraph"]
JUDGE_A = "deepseek-v4-pro"
JUDGE_B = "glm-5.2"

# Landis & Koch (1977) benchmarks, applied to weighted kappa.
_KAPPA_BANDS = [
    (0.20, "leve"),
    (0.40, "aceptable"),
    (0.60, "moderada"),
    (0.80, "sustancial"),
    (1.01, "casi perfecta"),
]

# Koo & Li (2016) benchmarks for ICC.
_ICC_BANDS = [
    (0.50, "pobre"),
    (0.75, "moderada"),
    (0.90, "buena"),
    (1.01, "excelente"),
]


def _band(value: float, bands: list[tuple[float, str]]) -> str:
    if value < 0:
        return "peor que el azar"
    for upper, label in bands:
        if value < upper:
            return label
    return bands[-1][1]


def load_paired_scores(
    checkpoint: dict, mode: str
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Extract score vectors for both judges on a mode, aligned by claim index.

    Claims where either judge failed (score == 0 sentinel) are dropped
    pairwise, since a failure carries no rating information.

    Returns (scores_a, scores_b, diagnostics).
    """
    entry_a = checkpoint.get(f"{mode}__{JUDGE_A}", {})
    entry_b = checkpoint.get(f"{mode}__{JUDGE_B}", {})

    by_idx_a = {s["idx"]: s["score"] for s in entry_a.get("scores", [])}
    by_idx_b = {s["idx"]: s["score"] for s in entry_b.get("scores", [])}

    shared = sorted(set(by_idx_a) & set(by_idx_b))
    paired = [(by_idx_a[i], by_idx_b[i]) for i in shared]
    valid = [(a, b) for a, b in paired if a > 0 and b > 0]

    diagnostics = {
        "n_judge_a": len(by_idx_a),
        "n_judge_b": len(by_idx_b),
        "n_shared": len(shared),
        "n_valid": len(valid),
        "n_dropped_failures": len(shared) - len(valid),
    }

    if not valid:
        return np.array([]), np.array([]), diagnostics

    arr = np.array(valid, dtype=float)
    return arr[:, 0], arr[:, 1], diagnostics


def weighted_kappa(a: np.ndarray, b: np.ndarray, weights: str) -> float | None:
    """Cohen's kappa on the raw integer scores with the given weighting."""
    from sklearn.metrics import cohen_kappa_score

    labels = list(range(1, 11))
    try:
        return float(
            cohen_kappa_score(
                a.astype(int), b.astype(int), labels=labels, weights=weights
            )
        )
    except Exception:
        return None


def icc_2_1(a: np.ndarray, b: np.ndarray) -> float | None:
    """ICC(2,1): two-way random effects, single rater, absolute agreement.

    Ratings matrix is n subjects x k=2 raters. Using the standard
    Shrout & Fleiss (1979) decomposition:

        ICC(2,1) = (MS_R - MS_E)
                   / (MS_R + (k-1)·MS_E + k·(MS_C - MS_E)/n)

    where MS_R is the between-subject mean square, MS_C the between-rater
    mean square, and MS_E the residual mean square.
    """
    ratings = np.column_stack([a, b])
    n, k = ratings.shape
    if n < 2:
        return None

    grand_mean = ratings.mean()
    subject_means = ratings.mean(axis=1)
    rater_means = ratings.mean(axis=0)

    ss_total = ((ratings - grand_mean) ** 2).sum()
    ss_rows = k * ((subject_means - grand_mean) ** 2).sum()
    ss_cols = n * ((rater_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = df_rows * df_cols
    if df_error <= 0:
        return None

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error

    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denom == 0:
        return None
    return float((ms_rows - ms_error) / denom)


def analyse_pair(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Compute the full agreement panel for one paired score vector."""
    from scipy import stats

    n = len(a)
    if n < 2:
        return {"note": "insufficient paired observations"}

    kappa_q = weighted_kappa(a, b, "quadratic")
    kappa_l = weighted_kappa(a, b, "linear")
    kappa_u = weighted_kappa(a, b, None)

    # Rank and linear correlation. Constant vectors make these undefined.
    if a.std() == 0 or b.std() == 0:
        rho = rho_p = pearson = pearson_p = None
    else:
        rho_res = stats.spearmanr(a, b)
        pearson_res = stats.pearsonr(a, b)
        rho = float(rho_res.statistic)
        rho_p = float(rho_res.pvalue)
        pearson = float(pearson_res.statistic)
        pearson_p = float(pearson_res.pvalue)

    icc = icc_2_1(a, b)

    diff = a - b
    bias = float(diff.mean())
    sd_diff = float(diff.std(ddof=1)) if n > 1 else 0.0

    # Exact agreement and near agreement (within one scale point).
    exact = float((a == b).mean())
    within_one = float((np.abs(diff) <= 1).mean())

    return {
        "n": n,
        "kappa_quadratic": None if kappa_q is None else round(kappa_q, 4),
        "kappa_quadratic_band": None if kappa_q is None else _band(kappa_q, _KAPPA_BANDS),
        "kappa_linear": None if kappa_l is None else round(kappa_l, 4),
        "kappa_unweighted": None if kappa_u is None else round(kappa_u, 4),
        "spearman_rho": None if rho is None else round(rho, 4),
        "spearman_p": None if rho_p is None else float(f"{rho_p:.3g}"),
        "pearson_r": None if pearson is None else round(pearson, 4),
        "pearson_p": None if pearson_p is None else float(f"{pearson_p:.3g}"),
        "icc_2_1": None if icc is None else round(icc, 4),
        "icc_band": None if icc is None else _band(icc, _ICC_BANDS),
        "mean_judge_a": round(float(a.mean()), 4),
        "mean_judge_b": round(float(b.mean()), 4),
        "sd_judge_a": round(float(a.std(ddof=1)), 4),
        "sd_judge_b": round(float(b.std(ddof=1)), 4),
        "calibration_bias": round(bias, 4),
        "sd_of_differences": round(sd_diff, 4),
        "loa_lower": round(bias - 1.96 * sd_diff, 4),
        "loa_upper": round(bias + 1.96 * sd_diff, 4),
        "exact_agreement": round(exact, 4),
        "agreement_within_1": round(within_one, 4),
    }


def _fmt(value: Any, width: int, decimals: int = 3) -> str:
    if value is None:
        return "—".rjust(width)
    if isinstance(value, float):
        return f"{value:>{width}.{decimals}f}"
    return str(value).rjust(width)


def print_report(per_mode: dict[str, dict], pooled: dict) -> None:
    print("=" * 78)
    print("CONCORDANCIA ENTRE JUECES — estadísticos apropiados para escala ordinal")
    print("=" * 78)
    print(f"Juez A: {JUDGE_A}   Juez B: {JUDGE_B}")
    print()

    header = (
        f"{'Modo':<12}{'n':>6}{'κ_quad':>9}{'ρ':>8}{'ICC':>8}"
        f"{'κ_sin_peso':>12}{'±1 pto':>9}{'sesgo':>8}"
    )
    print(header)
    print("-" * len(header))

    for mode in MODES:
        r = per_mode.get(mode)
        if not r or "note" in r:
            print(f"{mode:<12}{'sin datos':>6}")
            continue
        print(
            f"{mode:<12}"
            f"{r['n']:>6}"
            f"{_fmt(r['kappa_quadratic'], 9)}"
            f"{_fmt(r['spearman_rho'], 8)}"
            f"{_fmt(r['icc_2_1'], 8)}"
            f"{_fmt(r['kappa_unweighted'], 12)}"
            f"{_fmt(r['agreement_within_1'], 9)}"
            f"{_fmt(r['calibration_bias'], 8)}"
        )

    if pooled and "note" not in pooled:
        print("-" * len(header))
        print(
            f"{'AGREGADO':<12}"
            f"{pooled['n']:>6}"
            f"{_fmt(pooled['kappa_quadratic'], 9)}"
            f"{_fmt(pooled['spearman_rho'], 8)}"
            f"{_fmt(pooled['icc_2_1'], 8)}"
            f"{_fmt(pooled['kappa_unweighted'], 12)}"
            f"{_fmt(pooled['agreement_within_1'], 9)}"
            f"{_fmt(pooled['calibration_bias'], 8)}"
        )

    print()
    print("Leyenda")
    print("  κ_quad      Cohen's κ con pesos cuadráticos sobre scores 1-10 crudos.")
    print("  ρ           Spearman: concordancia en el ordenamiento, inmune a recalibración.")
    print("  ICC         ICC(2,1), efectos aleatorios de dos vías, acuerdo absoluto.")
    print("  κ_sin_peso  κ sin pesos sobre terciles — el estadístico original, para contraste.")
    print("  ±1 pto      Proporción de argumentos donde los jueces difieren en ≤1 punto.")
    print("  sesgo       Diferencia media A−B: desplazamiento sistemático de calibración.")
    print()

    if pooled and "note" not in pooled:
        print("Interpretación agregada")
        print(
            f"  κ ponderado: {pooled['kappa_quadratic']} "
            f"({pooled['kappa_quadratic_band']})"
        )
        print(f"  ICC(2,1):    {pooled['icc_2_1']} ({pooled['icc_band']})")
        if pooled["spearman_rho"] is not None:
            print(
                f"  Spearman ρ:  {pooled['spearman_rho']} (p={pooled['spearman_p']})"
            )
        print(
            f"  Calibración: {JUDGE_A} puntúa en promedio "
            f"{pooled['calibration_bias']:+.3f} puntos respecto de {JUDGE_B}."
        )
        print(
            f"  Límites de acuerdo (Bland-Altman): "
            f"[{pooled['loa_lower']}, {pooled['loa_upper']}]"
        )
        print(
            f"  Los jueces difieren en ≤1 punto en el "
            f"{pooled['agreement_within_1'] * 100:.1f}% de los argumentos."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute inter-judge agreement with ordinal-appropriate statistics"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Judge checkpoint produced by run_judge_smoke.py",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Where to write the agreement results",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1

    with open(args.checkpoint, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    per_mode: dict[str, dict] = {}
    pooled_a: list[np.ndarray] = []
    pooled_b: list[np.ndarray] = []

    for mode in MODES:
        a, b, diag = load_paired_scores(checkpoint, mode)
        if len(a) == 0:
            per_mode[mode] = {"note": "no paired scores", **diag}
            continue
        result = analyse_pair(a, b)
        result.update(diag)
        per_mode[mode] = result
        pooled_a.append(a)
        pooled_b.append(b)

    if pooled_a:
        all_a = np.concatenate(pooled_a)
        all_b = np.concatenate(pooled_b)
        pooled = analyse_pair(all_a, all_b)
    else:
        pooled = {"note": "no paired scores in any mode"}

    print_report(per_mode, pooled)

    payload = {
        "judge_a": JUDGE_A,
        "judge_b": JUDGE_B,
        "per_mode": per_mode,
        "pooled": pooled,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
