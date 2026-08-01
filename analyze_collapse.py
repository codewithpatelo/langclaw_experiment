"""Analysis of transcript-grounded context collapse across temporal windows.

Reads the audit produced by run_collapse_judge.py and answers three questions.

  1. Does fidelity to the record degrade as the debate progresses?
     Collapse incidence per temporal window, per mode, where an argument counts
     as collapsed if any failure mode is flagged with supporting evidence.

  2. Does fluency degrade alongside fidelity?
     If fluency stays flat while fidelity falls, the High-Functioning
     Compensation Effect is present in these data: the failure is invisible to
     quality scoring, which is precisely why the earlier per-argument campaign
     could not detect it.

  3. Does the activation regime change the collapse trajectory?
     Per-seed collapse rate is regressed on window index, giving one slope per
     (mode, seed), and EPR is compared against each condition with a paired
     Wilcoxon signed-rank test over shared seeds.

A flag is honoured only when the judge supplied a verbatim quotation, which
bounds false positives. Agreement between the two auditors is reported per
failure mode.

Usage
-----
    python analyze_collapse.py
    python analyze_collapse.py --judge deepseek-v4-pro
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).parent
DEFAULT_RESULTS = BASE_DIR / "collapse_judge_results.json"
DEFAULT_OUTPUT = BASE_DIR / "collapse_analysis.json"

MODES = ["epr", "epr_q", "epr_sham", "langgraph"]
N_WINDOWS = 5

# Internal keys are those emitted by the judging prompt and must stay stable so
# that existing checkpoints remain readable. Display labels use correct Spanish:
# "misatribucion" and "conflacion" are anglicisms absent from the RAE.
FAILURE_MODES = [
    "fabricacion",
    "misatribucion",
    "distorsion_objetivo",
    "amnesia",
    "conflacion",
]
LABELS = {
    "fabricacion": "fabricacion",
    "misatribucion": "atribucion erronea",
    "distorsion_objetivo": "distorsion del objetivo",
    "amnesia": "amnesia",
    "conflacion": "fusion indebida",
    "collapsed": "cualquier falla",
    "collapsed_any": "cualquier falla (un juez)",
    "fluidez": "fluidez",
}
MIN_EVIDENCE_CHARS = 12


def has_evidence(record: dict) -> bool:
    """A flag is credited only if the auditor quoted the offending text."""
    return len(str(record.get("evidencia", "")).strip()) >= MIN_EVIDENCE_CHARS


def collapsed(record: dict) -> bool:
    flags = record.get("flags", {})
    return any(flags.get(m) == 1 for m in FAILURE_MODES) and has_evidence(record)


def build_frame(payload: dict, judge: str) -> list[dict[str, Any]]:
    """Join audit verdicts with unit metadata, keeping successful audits only."""
    units = payload["units"]
    audits = payload["judges"].get(judge, {})
    rows: list[dict[str, Any]] = []
    for unit_id, meta in units.items():
        rec = audits.get(unit_id)
        if not rec or not rec.get("ok"):
            continue
        flags = rec.get("flags", {})
        ev = has_evidence(rec)
        row = {
            "mode": meta["mode"],
            "seed": meta["seed"],
            "window": meta["window"],
            "tick": meta["tick"],
            "n_prior": meta["n_prior"],
            "collapsed": int(collapsed(rec)),
            "fluidez": rec.get("fluidez", 0),
        }
        for m in FAILURE_MODES:
            row[m] = int(flags.get(m) == 1 and ev)
        rows.append(row)
    return rows


def rate_by_window(rows: list[dict], mode: str, field: str) -> tuple[list[float | None], list[int]]:
    rates: list[float | None] = []
    counts: list[int] = []
    for w in range(N_WINDOWS):
        vals = [r[field] for r in rows if r["mode"] == mode and r["window"] == w]
        counts.append(len(vals))
        rates.append(float(np.mean(vals)) if vals else None)
    return rates, counts


def per_seed_slope(rows: list[dict], mode: str, field: str) -> dict[int, float]:
    """Slope of per-window mean of `field` on window index, one value per seed."""
    by_seed: dict[int, dict[int, list[float]]] = {}
    for r in rows:
        if r["mode"] != mode:
            continue
        by_seed.setdefault(r["seed"], {}).setdefault(r["window"], []).append(r[field])

    slopes: dict[int, float] = {}
    for seed, windows in by_seed.items():
        pts = [(w, float(np.mean(v))) for w, v in sorted(windows.items()) if v]
        if len(pts) < 3:
            continue
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        if x.std() == 0:
            continue
        slopes[seed] = float(np.polyfit(x, y, 1)[0])
    return slopes


def paired_test(a: dict[int, float], b: dict[int, float]) -> dict[str, Any]:
    from scipy import stats

    seeds = sorted(a.keys() & b.keys())
    if len(seeds) < 6:
        return {"n_pairs": len(seeds), "note": "too few paired seeds"}
    x = np.array([a[s] for s in seeds])
    y = np.array([b[s] for s in seeds])
    if np.allclose(x, y):
        return {"n_pairs": len(seeds), "note": "identical"}
    res = stats.wilcoxon(x, y, alternative="two-sided")
    return {
        "n_pairs": len(seeds),
        "median_a": round(float(np.median(x)), 5),
        "median_b": round(float(np.median(y)), 5),
        "p_value": float(f"{res.pvalue:.3g}"),
        "seeds_a_lower": int((x < y).sum()),
    }


def trend_test(slopes: dict[int, float]) -> dict[str, Any]:
    """Is the per-seed slope distribution centred away from zero?"""
    from scipy import stats

    v = list(slopes.values())
    if len(v) < 6:
        return {"n": len(v), "note": "too few seeds"}
    res = stats.wilcoxon(v, alternative="two-sided")
    return {
        "n": len(v),
        "median_slope": round(float(np.median(v)), 5),
        "p_value": float(f"{res.pvalue:.3g}"),
        "seeds_positive": int(sum(1 for x in v if x > 0)),
    }


def _f(v: float | None, w: int, d: int = 3) -> str:
    return "—".rjust(w) if v is None else f"{v:>{w}.{d}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse transcript-grounded collapse")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judge", type=str, default=None)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: no existe {args.results}. Corre primero run_collapse_judge.py")
        return 1
    with open(args.results, "r", encoding="utf-8") as f:
        payload = json.load(f)

    judges = list(payload["judges"].keys())
    print(f"Auditores disponibles: {judges}")

    frames = {j: build_frame(payload, j) for j in judges}
    for j, rows in frames.items():
        print(f"  {j}: {len(rows)} auditorias validas")

    # Consensus frame: an argument counts as collapsed when both auditors agree,
    # which is the conservative reading.
    primary = args.judge or judges[0]
    rows = frames[primary]

    if len(judges) >= 2:
        a, b = judges[0], judges[1]
        idx_a = {
            (r["mode"], r["seed"], r["window"], r["tick"]): r for r in frames[a]
        }
        idx_b = {
            (r["mode"], r["seed"], r["window"], r["tick"]): r for r in frames[b]
        }
        shared = sorted(idx_a.keys() & idx_b.keys())
        print(f"\nUnidades auditadas por ambos: {len(shared)}")
        print(f"{'Modo de falla':<26}{'tasa A':>9}{'tasa B':>9}{'acuerdo':>10}{'ambos':>8}")
        print("-" * 62)
        agreement: dict[str, Any] = {}
        for m in FAILURE_MODES + ["collapsed"]:
            va = np.array([idx_a[k][m] for k in shared])
            vb = np.array([idx_b[k][m] for k in shared])
            agree = float((va == vb).mean())
            both = int(((va == 1) & (vb == 1)).sum())
            agreement[m] = {
                "rate_judge_a": round(float(va.mean()), 4),
                "rate_judge_b": round(float(vb.mean()), 4),
                "raw_agreement": round(agree, 4),
                "both_flagged": both,
            }
            print(
                f"{LABELS.get(m, m):<26}{va.mean():>9.3f}{vb.mean():>9.3f}"
                f"{agree:>10.3f}{both:>8}"
            )

        consensus = []
        for k in shared:
            ra, rb = idx_a[k], idx_b[k]
            row = {
                "mode": ra["mode"], "seed": ra["seed"], "window": ra["window"],
                "tick": ra["tick"], "n_prior": ra["n_prior"],
                "collapsed": int(ra["collapsed"] == 1 and rb["collapsed"] == 1),
                "collapsed_any": int(ra["collapsed"] == 1 or rb["collapsed"] == 1),
                "fluidez": (ra["fluidez"] + rb["fluidez"]) / 2,
            }
            for m in FAILURE_MODES:
                row[m] = int(ra[m] == 1 and rb[m] == 1)
            consensus.append(row)
    else:
        agreement = {}
        consensus = rows

    panels: dict[str, Any] = {}

    for label, frame, field in [
        ("CONSENSO (ambos auditores marcan la falla)", consensus, "collapsed"),
        ("CUALQUIER AUDITOR marca la falla", consensus, "collapsed_any"),
        ("FLUIDEZ (calidad de superficie)", consensus, "fluidez"),
    ]:
        if field not in frame[0]:
            continue
        print()
        print("=" * 78)
        print(f"{label}")
        print("=" * 78)
        header = f"{'Ventana':<14}" + "".join(f"{m:>15}" for m in MODES)
        print(header)
        print("-" * len(header))

        panel: dict[str, Any] = {"by_window": {}, "counts": {}}
        for w in range(N_WINDOWS):
            line = f"{f'w{w} (p{w*16}-{w*16+15})':<14}"
            for mode in MODES:
                r, c = rate_by_window(frame, mode, field)
                line += _f(r[w], 15)
            print(line)
        for mode in MODES:
            r, c = rate_by_window(frame, mode, field)
            panel["by_window"][mode] = [None if x is None else round(x, 4) for x in r]
            panel["counts"][mode] = c

        print("-" * len(header))
        line = f"{'n auditados':<14}"
        for mode in MODES:
            line += f"{sum(panel['counts'][mode]):>15}"
        print(line)

        slopes = {m: per_seed_slope(frame, m, field) for m in MODES}
        line = f"{'pendiente':<14}"
        for mode in MODES:
            v = list(slopes[mode].values())
            line += _f(float(np.median(v)) if v else None, 15, 4)
        print(line)

        line = f"{'p (pend=0)':<14}"
        trends: dict[str, Any] = {}
        for mode in MODES:
            t = trend_test(slopes[mode])
            trends[mode] = t
            line += (
                f"{t['p_value']:>15.3g}" if "p_value" in t else f"{'—':>15}"
            )
        print(line)
        panel["trend_tests"] = trends
        panel["median_slope"] = {
            m: (round(float(np.median(list(slopes[m].values()))), 6) if slopes[m] else None)
            for m in MODES
        }

        others = [m for m in MODES if m != "epr" and slopes[m]]
        alpha = 0.05 / max(1, len(others))
        print(f"\nWilcoxon pareado vs. EPR (Bonferroni alpha={alpha:.4f})")
        tests: dict[str, Any] = {}
        for mode in others:
            t = paired_test(slopes["epr"], slopes[mode])
            if "note" in t:
                print(f"  epr vs {mode:<12} {t['note']}")
                continue
            t["significant"] = bool(t["p_value"] < alpha)
            tests[f"epr_vs_{mode}"] = t
            print(
                f"  epr vs {mode:<12} {t['median_a']:>+9.4f} vs {t['median_b']:>+9.4f}"
                f"  p={t['p_value']:<10.3g}"
                f"{'SIG' if t['significant'] else 'no sig'}"
                f"  ({t['seeds_a_lower']}/{t['n_pairs']} semillas EPR menor)"
            )
        panel["paired_tests_vs_epr"] = tests
        panels[field] = panel

    # Breakdown of individual failure modes, consensus reading.
    print()
    print("=" * 78)
    print("DESGLOSE POR MODO DE FALLA (consenso), tasa por ventana")
    print("=" * 78)
    breakdown: dict[str, Any] = {}
    for fm in FAILURE_MODES:
        print(f"\n{LABELS.get(fm, fm)}")
        header = f"{'  Ventana':<14}" + "".join(f"{m:>15}" for m in MODES)
        print(header)
        breakdown[fm] = {}
        for w in range(N_WINDOWS):
            line = f"{f'  w{w}':<14}"
            for mode in MODES:
                r, _ = rate_by_window(consensus, mode, fm)
                line += _f(r[w], 15)
            print(line)
        for mode in MODES:
            r, _ = rate_by_window(consensus, mode, fm)
            breakdown[fm][mode] = [None if x is None else round(x, 4) for x in r]

    payload_out = {
        "primary_judge": primary,
        "inter_judge_agreement": agreement,
        "panels": panels,
        "failure_mode_breakdown": breakdown,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload_out, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
