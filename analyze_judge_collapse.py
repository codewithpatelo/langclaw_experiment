"""Context collapse measured by blind LLM judges instead of the internal signal g.

Why this analysis
-----------------
The context-collapse result was previously computed on g, the satiation term of
the homeostatic loop. Because g is the quantity EPR regulates against, using it
to score EPR is circular, and a per-argument check showed that g carries almost
no shared variance with blind judge scores once degenerate arguments are
excluded. The temporal-degradation claim therefore has to be re-established on a
measure that is external to the mechanism.

Judge scores qualify: two LLM judges rated every debate argument blind to
condition. Each argument carries a tick, so the scores can be binned into the
same temporal windows used before, and the degradation slope can be estimated
per run and compared across modes.

What is computed
----------------
  1. Mean judge score per temporal window, per mode.
  2. Degradation from first to last window (percentage of the first window).
  3. Per-run OLS slope of judge score on tick, giving one slope per
     (mode, seed). This is the unit of analysis for inference.
  4. Wilcoxon signed-rank on paired per-seed slopes, EPR against each other
     condition, Bonferroni-corrected. Paired because all modes share seeds.
  5. The same panel computed on g, so the two measures sit side by side.

Usage
-----
    python analyze_judge_collapse.py
    python analyze_judge_collapse.py --windows 4
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
DEFAULT_OUTPUT = BASE_DIR / "judge_collapse_results.json"

MODES = ["epr", "epr_q", "epr_sham", "langgraph"]
LOTS = ["lot1", "lot2", "lot3", "lot4"]
SEEDS_PER_LOT = 5
JUDGE_A = "deepseek-v4-pro"
JUDGE_B = "glm-5.2"
TOTAL_TICKS = 80


def get_completed_seeds(lot: str) -> list[int]:
    ck_path = OUTPUT_BASE / lot / "benchmark_checkpoint.json"
    if not ck_path.exists():
        return []
    with open(ck_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return sorted(set(e["_seed"] for e in entries))[:SEEDS_PER_LOT]


def load_argument_index(mode: str) -> dict[str, dict[str, Any]]:
    """Map node_id -> {tick, seed, lot, g} for every DEBATE argument of a mode."""
    safe_mode = mode.replace("-", "_")
    index: dict[str, dict[str, Any]] = {}
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
                    if not node_id:
                        continue
                    index[node_id] = {
                        "tick": int(entry.get("tick", 0)),
                        "seed": seed,
                        "lot": lot,
                        "g": float(entry.get("delta_phi", 0.0)),
                    }
    return index


def load_judge_means(checkpoint: dict, mode: str) -> dict[str, float]:
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


def build_records(checkpoint: dict, mode: str) -> list[dict[str, Any]]:
    """One record per argument that has both a tick and a pair of judge scores."""
    index = load_argument_index(mode)
    judges = load_judge_means(checkpoint, mode)
    records = []
    for node_id, score in judges.items():
        meta = index.get(node_id)
        if meta is None:
            continue
        records.append({"node_id": node_id, "judge": score, **meta})
    return records


def window_of(tick: int, n_windows: int) -> int:
    width = TOTAL_TICKS / n_windows
    return min(int(tick / width), n_windows - 1)


def window_means(
    records: list[dict], field: str, n_windows: int
) -> tuple[list[float | None], list[int]]:
    buckets: list[list[float]] = [[] for _ in range(n_windows)]
    for r in records:
        buckets[window_of(r["tick"], n_windows)].append(r[field])
    means = [float(np.mean(b)) if b else None for b in buckets]
    counts = [len(b) for b in buckets]
    return means, counts


def degradation_pct(means: list[float | None]) -> float | None:
    present = [m for m in means if m is not None]
    if len(present) < 2 or present[0] == 0:
        return None
    return 100.0 * (present[0] - present[-1]) / present[0]


def per_seed_slopes(records: list[dict], field: str) -> dict[int, float]:
    """OLS slope of `field` on tick, computed separately for each seed."""
    by_seed: dict[int, list[tuple[int, float]]] = {}
    for r in records:
        by_seed.setdefault(r["seed"], []).append((r["tick"], r[field]))

    slopes: dict[int, float] = {}
    for seed, pairs in by_seed.items():
        if len(pairs) < 3:
            continue
        x = np.array([p[0] for p in pairs], dtype=float)
        y = np.array([p[1] for p in pairs], dtype=float)
        if x.std() == 0:
            continue
        slopes[seed] = float(np.polyfit(x, y, 1)[0])
    return slopes


def paired_wilcoxon(
    slopes_a: dict[int, float], slopes_b: dict[int, float]
) -> dict[str, Any]:
    """Wilcoxon signed-rank on slopes paired by seed.

    A less negative slope means slower degradation. The alternative is
    two-sided; direction is reported through the median difference.
    """
    from scipy import stats

    seeds = sorted(slopes_a.keys() & slopes_b.keys())
    if len(seeds) < 6:
        return {"n_pairs": len(seeds), "note": "too few paired seeds"}

    a = np.array([slopes_a[s] for s in seeds])
    b = np.array([slopes_b[s] for s in seeds])
    diff = a - b
    if np.allclose(diff, 0):
        return {"n_pairs": len(seeds), "note": "identical slopes"}

    res = stats.wilcoxon(a, b, alternative="two-sided")
    n_a_better = int((a > b).sum())
    return {
        "n_pairs": len(seeds),
        "median_slope_a": round(float(np.median(a)), 5),
        "median_slope_b": round(float(np.median(b)), 5),
        "median_difference": round(float(np.median(diff)), 5),
        "statistic": float(res.statistic),
        "p_value": float(f"{res.pvalue:.3g}"),
        "seeds_where_a_degrades_less": n_a_better,
        "seeds_total": len(seeds),
    }


def _fmt(v: float | None, w: int, d: int = 3) -> str:
    return "—".rjust(w) if v is None else f"{v:>{w}.{d}f}"


def report_field(
    label: str,
    field: str,
    records_by_mode: dict[str, list[dict]],
    n_windows: int,
) -> dict[str, Any]:
    width = TOTAL_TICKS // n_windows
    print()
    print("=" * 82)
    print(f"COLAPSO DE CONTEXTO SEGUN {label}")
    print("=" * 82)

    header = f"{'Ventana':<16}" + "".join(f"{m:>14}" for m in MODES)
    print(header)
    print("-" * len(header))

    means_by_mode: dict[str, list[float | None]] = {}
    counts_by_mode: dict[str, list[int]] = {}
    for mode in MODES:
        means, counts = window_means(records_by_mode.get(mode, []), field, n_windows)
        means_by_mode[mode] = means
        counts_by_mode[mode] = counts

    for w in range(n_windows):
        lo, hi = w * width, (w + 1) * width - 1
        row = f"{f'Pulsos {lo}-{hi}':<16}"
        for mode in MODES:
            row += _fmt(means_by_mode[mode][w], 14)
        print(row)

    row = f"{'Degradacion %':<16}"
    degradations: dict[str, float | None] = {}
    for mode in MODES:
        d = degradation_pct(means_by_mode[mode])
        degradations[mode] = d
        row += _fmt(d, 14, 2)
    print("-" * len(header))
    print(row)

    row = f"{'n argumentos':<16}"
    for mode in MODES:
        row += f"{sum(counts_by_mode[mode]):>14}"
    print(row)

    # Per-seed slopes and paired inference against EPR.
    slopes = {m: per_seed_slopes(records_by_mode.get(m, []), field) for m in MODES}

    row = f"{'Pendiente med.':<16}"
    for mode in MODES:
        s = slopes[mode]
        row += _fmt(float(np.median(list(s.values()))) if s else None, 14, 5)
    print(row)

    others = [m for m in MODES if m != "epr" and slopes[m]]
    alpha = 0.05 / max(1, len(others))
    print()
    print(f"Wilcoxon pareado por semilla, EPR vs. cada condicion (Bonferroni alpha={alpha:.4f})")
    print(
        f"{'Comparacion':<26}{'pend. EPR':>12}{'pend. otro':>12}"
        f"{'p':>11}{'sig':>6}{'semillas EPR mejor':>20}"
    )
    print("-" * 87)

    tests: dict[str, Any] = {}
    for mode in others:
        t = paired_wilcoxon(slopes["epr"], slopes[mode])
        tests[f"epr_vs_{mode}"] = t
        if "note" in t:
            print(f"{'epr vs ' + mode:<26}{t['note']}")
            continue
        t["alpha_bonferroni"] = round(alpha, 4)
        t["significant"] = bool(t["p_value"] < alpha)
        ratio = f"{t['seeds_where_a_degrades_less']}/{t['seeds_total']}"
        print(
            f"{'epr vs ' + mode:<26}"
            f"{t['median_slope_a']:>12.5f}{t['median_slope_b']:>12.5f}"
            f"{t['p_value']:>11.3g}{('SI' if t['significant'] else 'no'):>6}"
            f"{ratio:>20}"
        )

    return {
        "window_means": {m: means_by_mode[m] for m in MODES},
        "window_counts": {m: counts_by_mode[m] for m in MODES},
        "degradation_pct": degradations,
        "median_slope": {
            m: (round(float(np.median(list(slopes[m].values()))), 6) if slopes[m] else None)
            for m in MODES
        },
        "per_seed_slopes": {m: {str(k): round(v, 6) for k, v in slopes[m].items()} for m in MODES},
        "paired_tests_vs_epr": tests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Context collapse measured by blind LLM judges"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", type=int, default=5)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1
    with open(args.checkpoint, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    records_by_mode = {m: build_records(checkpoint, m) for m in MODES}
    for mode, recs in records_by_mode.items():
        print(f"{mode:<12} {len(recs)} argumentos con tick y puntaje de ambos jueces")

    judge_panel = report_field("LOS JUECES", "judge", records_by_mode, args.windows)
    g_panel = report_field("g (senal de control)", "g", records_by_mode, args.windows)

    payload = {
        "n_windows": args.windows,
        "judge": judge_panel,
        "g": g_panel,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
