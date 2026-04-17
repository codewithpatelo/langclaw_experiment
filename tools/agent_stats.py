"""Per-seed agent participation and final-deficit statistics.

Computes, for each (mode, seed):
  - per-agent debate counts
  - sigma_s (std dev across agents)
  - n_silenced (agents with 0 debates)
  - faction asymmetry (GOV vs OPP final deficits, when present)

Outputs aggregate mean +/- std across seeds.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_logs(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["logs"] if isinstance(data, dict) and "logs" in data else data


def _per_agent_debates(logs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in logs:
        if entry.get("action") == "DEBATE" and entry.get("node_id"):
            counts[entry["agent_id"]] = counts.get(entry["agent_id"], 0) + 1
    return counts


def _all_agents(logs: list[dict[str, Any]]) -> set[str]:
    return {entry["agent_id"] for entry in logs if "agent_id" in entry}


def analyze(path: Path) -> dict[str, Any]:
    logs = _load_logs(path)
    agents = sorted(_all_agents(logs))
    counts = _per_agent_debates(logs)
    counts_full = [counts.get(a, 0) for a in agents]
    n_silenced = sum(1 for c in counts_full if c == 0)
    sigma_s = statistics.stdev(counts_full) if len(counts_full) > 1 else 0.0
    max_share = max(counts_full) / sum(counts_full) if sum(counts_full) > 0 else 0.0

    final_deficits: dict[str, float] = {}
    for entry in reversed(logs):
        if "final_deficits" in entry:
            final_deficits = entry["final_deficits"]
            break
    if not final_deficits:
        for entry in reversed(logs):
            if entry.get("deficit") is not None:
                final_deficits.setdefault(entry["agent_id"], float(entry["deficit"]))
                if len(final_deficits) >= len(agents):
                    break

    gov = [v for k, v in final_deficits.items() if k.startswith("GOV")]
    opp = [v for k, v in final_deficits.items() if k.startswith("OPP")]

    return {
        "n_agents": len(agents),
        "total_debates": sum(counts_full),
        "sigma_s": round(sigma_s, 3),
        "n_silenced": n_silenced,
        "max_share": round(max_share, 3),
        "mean_def_gov": round(statistics.fmean(gov), 3) if gov else None,
        "mean_def_opp": round(statistics.fmean(opp), 3) if opp else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="benchmark_results_v7")
    parser.add_argument("--seeds", nargs="*", type=int, default=[7, 17, 42, 123])
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    rows: dict[str, list[dict[str, Any]]] = {"hrrl": [], "langgraph": []}

    for s in args.seeds:
        for mode in ("hrrl", "langgraph"):
            p = logs_dir / f"logs_{mode}_seed{s}.json"
            if not p.exists():
                continue
            r = analyze(p)
            r["seed"] = s
            r["mode"] = mode
            rows[mode].append(r)

    print("seed  mode       totalD  sigma_s  silenced  max_share  defGOV  defOPP")
    for mode in ("hrrl", "langgraph"):
        for r in rows[mode]:
            print(
                f"{r['seed']:>4}  {mode:<10} {r['total_debates']:>5}  "
                f"{r['sigma_s']:>6.2f}  {r['n_silenced']:>7}   {r['max_share']:>7.3f}  "
                f"{(r['mean_def_gov'] if r['mean_def_gov'] is not None else 0):>6.2f}  "
                f"{(r['mean_def_opp'] if r['mean_def_opp'] is not None else 0):>6.2f}"
            )

    print("\nAggregate (mean +/- std) across seeds:")
    for mode in ("hrrl", "langgraph"):
        sigmas = [r["sigma_s"] for r in rows[mode]]
        silenced = [r["n_silenced"] for r in rows[mode]]
        max_share = [r["max_share"] for r in rows[mode]]
        if not sigmas:
            continue
        print(
            f"  {mode:<10} sigma_s = {statistics.fmean(sigmas):.2f} +/- "
            f"{statistics.stdev(sigmas):.2f} | "
            f"silenced = {statistics.fmean(silenced):.2f} +/- "
            f"{statistics.stdev(silenced):.2f} | "
            f"max_share = {statistics.fmean(max_share):.3f} +/- "
            f"{statistics.stdev(max_share):.3f}"
        )


if __name__ == "__main__":
    main()
