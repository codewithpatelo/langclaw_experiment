"""Construct validity of the quality signal g against blind LLM judge scores.

Two questions are answered, both from data already on disk (no API calls):

  Q1. Per-argument convergent validity.
      For every judged argument we hold two independent measurements: the
      structural signal g computed from the argument graph, and the mean score
      of two blind LLM judges. If g is a valid proxy for argument quality the
      two must correlate positively at the level of the individual argument.
      This is the decisive test; mode-level averages cannot answer it, because
      aggregation can invert a within-group relationship (Simpson's paradox).

  Q2. Do judge scores differ across modes?
      Mann-Whitney U (two-sided) on per-argument judge means, EPR against each
      other condition, Bonferroni-corrected. Also reports Cliff's delta as a
      non-parametric effect size, since with n~900 per mode trivial differences
      reach significance.

Usage
-----
    python analyze_g_validity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).parent
OUTPUT_BASE = BASE_DIR / "experiment_results"
DEFAULT_CHECKPOINT = BASE_DIR / "judge_smoke_checkpoint.json"
DEFAULT_OUTPUT = BASE_DIR / "g_validity_results.json"

MODES = ["epr", "epr_q", "epr_sham", "langgraph"]
LOTS = ["lot1", "lot2", "lot3", "lot4"]
SEEDS_PER_LOT = 5
JUDGE_A = "deepseek-v4-pro"
JUDGE_B = "glm-5.2"


def get_completed_seeds(lot: str) -> list[int]:
    ck_path = OUTPUT_BASE / lot / "benchmark_checkpoint.json"
    if not ck_path.exists():
        return []
    with open(ck_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return sorted(set(e["_seed"] for e in entries))[:SEEDS_PER_LOT]


def load_g_by_node(mode: str) -> dict[str, float]:
    """Map node_id -> g for every DEBATE argument of a mode, across all lots.

    node_id is globally unique per run, so entries never collide across seeds.
    """
    safe_mode = mode.replace("-", "_")
    g_by_node: dict[str, float] = {}
    for lot in LOTS:
        for seed in get_completed_seeds(lot):
            path = OUTPUT_BASE / lot / f"logs_{safe_mode}_seed{seed}.json"
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    if entry.get("action") != "DEBATE":
                        continue
                    node_id = entry.get("node_id")
                    if node_id:
                        g_by_node[node_id] = float(entry.get("delta_phi", 0.0))
    return g_by_node


def load_judge_means(checkpoint: dict, mode: str) -> dict[str, float]:
    """Map node_id -> mean judge score, keeping only arguments both judges rated."""
    a = {
        s["node_id"]: s["score"]
        for s in checkpoint.get(f"{mode}__{JUDGE_A}", {}).get("scores", [])
        if s["score"] > 0 and s.get("node_id")
    }
    b = {
        s["node_id"]: s["score"]
        for s in checkpoint.get(f"{mode}__{JUDGE_B}", {}).get("scores", [])
        if s["score"] > 0 and s.get("node_id")
    }
    return {nid: (a[nid] + b[nid]) / 2 for nid in a.keys() & b.keys()}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta via the Mann-Whitney U identity: d = 2U/(nm) - 1.

    Sign convention: positive means x tends to exceed y.
    """
    from scipy import stats

    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


def _effect_label(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "despreciable"
    if a < 0.33:
        return "pequeno"
    if a < 0.474:
        return "mediano"
    return "grande"


def _corr_pair(g: np.ndarray, judge: np.ndarray) -> dict[str, Any]:
    from scipy import stats

    if len(g) < 3 or g.std() == 0 or judge.std() == 0:
        return {"note": "insufficient variance"}
    rho = stats.spearmanr(g, judge)
    r = stats.pearsonr(g, judge)
    return {
        "spearman_rho": round(float(rho.statistic), 4),
        "spearman_p": float(f"{rho.pvalue:.3g}"),
        "pearson_r": round(float(r.statistic), 4),
        "pearson_p": float(f"{r.pvalue:.3g}"),
    }


def correlate(g: np.ndarray, judge: np.ndarray) -> dict[str, Any]:
    """Correlate g with judge scores, before and after removing degenerate cases.

    g == 0 marks an argument that scored zero on every structural component.
    These are graph-level failures (no engagement, no novelty, no diversity),
    not merely low-quality arguments, and the judges rate them near the floor
    of the scale. A handful of such points sits far from the bulk of the data in
    both variables simultaneously, which is exactly the configuration that
    inflates Pearson's r while leaving rank correlation untouched. The reported
    validity coefficient must therefore be the one computed on non-degenerate
    arguments; the full-sample figure is retained only for contrast.
    """
    n = len(g)
    if n < 3:
        return {"n": n, "note": "insufficient data"}

    mask = g > 0
    n_zero = int((~mask).sum())

    out: dict[str, Any] = {
        "n": n,
        "n_degenerate_g_zero": n_zero,
        "pct_degenerate": round(100.0 * n_zero / n, 2),
        "mean_g": round(float(g.mean()), 4),
        "median_g": round(float(np.median(g)), 4),
        "mean_judge": round(float(judge.mean()), 4),
        "median_judge": round(float(np.median(judge)), 4),
        "judge_mean_if_g_zero": (
            round(float(judge[~mask].mean()), 4) if n_zero else None
        ),
        "judge_mean_if_g_positive": (
            round(float(judge[mask].mean()), 4) if mask.any() else None
        ),
        "full_sample": _corr_pair(g, judge),
    }

    if mask.sum() >= 3:
        out["excluding_degenerate"] = _corr_pair(g[mask], judge[mask])
        out["n_excluding_degenerate"] = int(mask.sum())
        out["mean_g_excluding_degenerate"] = round(float(g[mask].mean()), 4)
    else:
        out["excluding_degenerate"] = {"note": "too few non-degenerate points"}

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test construct validity of g against blind judge scores"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from scipy import stats

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1
    with open(args.checkpoint, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    paired: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_mode: dict[str, dict] = {}

    print("=" * 88)
    print("Q1 · VALIDEZ CONVERGENTE POR ARGUMENTO  (g vs. media de jueces)")
    print("=" * 88)
    print("  Muestra completa           |  Excluyendo casos degenerados (g=0)")
    print(
        f"{'Modo':<12}{'n':>6}{'rho':>8}{'r':>8}"
        f"{'   |':>4}{'n':>7}{'rho':>8}{'p(rho)':>11}{'r':>8}{'p(r)':>11}{'g=0':>6}"
    )
    print("-" * 88)

    def _row(label: str, res: dict) -> None:
        fs = res.get("full_sample", {})
        ex = res.get("excluding_degenerate", {})
        if "note" in fs or "note" in ex:
            print(f"{label:<12}{res.get('n', 0):>6}  {fs.get('note') or ex.get('note')}")
            return
        print(
            f"{label:<12}{res['n']:>6}{fs['spearman_rho']:>8.3f}{fs['pearson_r']:>8.3f}"
            f"{'   |':>4}{res['n_excluding_degenerate']:>7}"
            f"{ex['spearman_rho']:>8.3f}{ex['spearman_p']:>11.3g}"
            f"{ex['pearson_r']:>8.3f}{ex['pearson_p']:>11.3g}"
            f"{res['n_degenerate_g_zero']:>6}"
        )

    for mode in MODES:
        g_map = load_g_by_node(mode)
        j_map = load_judge_means(checkpoint, mode)
        shared = sorted(g_map.keys() & j_map.keys())
        if not shared:
            per_mode[mode] = {"note": "no overlap between logs and judge scores"}
            print(f"{mode:<12}{'sin datos':>6}")
            continue

        g = np.array([g_map[n] for n in shared], dtype=float)
        j = np.array([j_map[n] for n in shared], dtype=float)
        paired[mode] = (g, j)

        res = correlate(g, j)
        res["n_g_logged"] = len(g_map)
        res["n_judged"] = len(j_map)
        per_mode[mode] = res
        _row(mode, res)

    pooled: dict[str, Any] = {"note": "no data"}
    if paired:
        all_g = np.concatenate([g for g, _ in paired.values()])
        all_j = np.concatenate([j for _, j in paired.values()])
        pooled = correlate(all_g, all_j)
        print("-" * 88)
        _row("AGREGADO", pooled)

        print()
        print("Diagnostico de los casos degenerados")
        for mode in MODES:
            r = per_mode.get(mode, {})
            if r.get("judge_mean_if_g_zero") is None:
                continue
            print(
                f"  {mode:<11} g=0 en {r['n_degenerate_g_zero']:>3} argumentos "
                f"({r['pct_degenerate']:>4.1f}%) — juez medio {r['judge_mean_if_g_zero']:.2f} "
                f"vs {r['judge_mean_if_g_positive']:.2f} en el resto"
            )

    # ── Q2: do judge scores differ across modes? ───────────────────────────
    print()
    print("=" * 74)
    print("Q2 · SCORES DE JUECES ENTRE MODOS  (Mann-Whitney bilateral vs. EPR)")
    print("=" * 74)

    comparisons: dict[str, dict] = {}
    if "epr" in paired:
        ref = paired["epr"][1]
        others = [m for m in MODES if m != "epr" and m in paired]
        alpha = 0.05 / max(1, len(others))

        print(f"Bonferroni: alpha = 0.05/{len(others)} = {alpha:.4f}")
        print()
        print(
            f"{'Comparacion':<26}{'media':>9}{'vs EPR':>9}{'p':>12}"
            f"{'sig':>6}{'delta':>8}{'efecto':>14}"
        )
        print("-" * 74)
        print(f"{'epr (referencia)':<26}{ref.mean():>9.3f}")

        for mode in others:
            other = paired[mode][1]
            u = stats.mannwhitneyu(other, ref, alternative="two-sided")
            d = cliffs_delta(other, ref)
            sig = u.pvalue < alpha
            comparisons[f"{mode}_vs_epr"] = {
                "n_mode": len(other),
                "n_epr": len(ref),
                "mean_mode": round(float(other.mean()), 4),
                "mean_epr": round(float(ref.mean()), 4),
                "mean_diff": round(float(other.mean() - ref.mean()), 4),
                "u_statistic": float(u.statistic),
                "p_value": float(f"{u.pvalue:.3g}"),
                "alpha_bonferroni": round(alpha, 4),
                "significant": bool(sig),
                "cliffs_delta": round(d, 4),
                "effect_size": _effect_label(d),
            }
            print(
                f"{mode + ' vs epr':<26}{other.mean():>9.3f}"
                f"{other.mean() - ref.mean():>+9.3f}{u.pvalue:>12.3g}"
                f"{('SI' if sig else 'no'):>6}{d:>8.3f}{_effect_label(d):>14}"
            )

    print()
    print("Notas")
    print("  rho    Spearman por argumento entre g y la media de los dos jueces.")
    print("  delta  Cliff's delta: 0 = distribuciones intercambiables, +-1 = separacion total.")
    print("         Umbrales: <0.147 despreciable, <0.33 pequeno, <0.474 mediano.")

    payload = {
        "q1_per_argument_validity": {"per_mode": per_mode, "pooled": pooled},
        "q2_judge_scores_across_modes": comparisons,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
