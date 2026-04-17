"""Volume-matched analysis (Fase 0B).

For each seed in {7, 17, 42, 123, 256} that has both HRRL and LangGraph logs:
  1. Count N_LG = total DEBATE turns in LangGraph.
  2. Truncate the HRRL trace to the first K = N_LG DEBATE turns
     (everything that happens up to and including the K-th DEBATE).
  3. Reconstruct the AAF subgraph from the truncated DEBATE turns.
  4. Recompute: AAF acceptance ratio, defeat cycles, dialectical completeness,
     PRR_G, mean Δφ*, and slope of acceptance ratio.
  5. Export both per-seed metrics and a summary CSV.

Outputs:
  tools/volume_matched_results.csv          -- per-seed numbers
  tools/volume_matched_summary.json         -- aggregated stats

Usage:
    python tools/volume_matched_analysis.py
    python tools/volume_matched_analysis.py --seeds 7 17 42
    python tools/volume_matched_analysis.py --logs-dir benchmark_results_v7
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log loading and filtering
# ---------------------------------------------------------------------------


def _load_logs(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "logs" in data:
        return data["logs"]
    return data


def _debate_entries(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return DEBATE entries in chronological order (tick, agent_id stable)."""
    debates = [l for l in logs if l.get("action") == "DEBATE" and l.get("node_id")]
    # Stable order: by tick then agent_id (matches how the env adds nodes).
    debates.sort(key=lambda l: (l["tick"], l["agent_id"]))
    return debates


def _truncate_to_k_debates(logs: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Keep all events up to and including the k-th DEBATE turn (chronological)."""
    if k <= 0:
        return []

    debates = _debate_entries(logs)
    if len(debates) <= k:
        return list(logs)

    # The (k-1)-th debate marks the cut-off tick.
    cutoff_tick = debates[k - 1]["tick"]
    cutoff_agents_at_tick = {d["agent_id"] for d in debates[:k] if d["tick"] == cutoff_tick}

    truncated: list[dict[str, Any]] = []
    debate_count = 0
    for entry in sorted(logs, key=lambda l: (l["tick"], l["agent_id"])):
        if entry["tick"] < cutoff_tick:
            truncated.append(entry)
            if entry.get("action") == "DEBATE" and entry.get("node_id"):
                debate_count += 1
            continue
        if entry["tick"] > cutoff_tick:
            break
        # entry tick == cutoff_tick: only include events whose agent debated by k.
        truncated.append(entry)
        if entry.get("action") == "DEBATE" and entry.get("node_id"):
            if entry["agent_id"] in cutoff_agents_at_tick and debate_count < k:
                debate_count += 1

    return truncated


def _select_last_k_debates(logs: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Return the last K DEBATE entries (chronological) as a bare debate list.

    This complements _truncate_to_k_debates: instead of slicing by the first K
    debates (early timeline, short-context), this slices the last K (late
    timeline, full-context). Used to neutralise the context-length confound
    flagged by REV5 in volume-matched comparisons.

    Unlike _truncate_to_k_debates this does NOT preserve non-DEBATE events
    interleaved in the original logs; AAF and PRR_G are reconstructed from
    the returned DEBATE subset alone.
    """
    if k <= 0:
        return []
    debates = _debate_entries(logs)
    if len(debates) <= k:
        return list(debates)
    return debates[-k:]


# ---------------------------------------------------------------------------
# AAF / PRR_G / Δφ* recomputation from filtered debate turns
# ---------------------------------------------------------------------------


def _build_subgraph(debates: list[dict[str, Any]]) -> nx.DiGraph:
    g = nx.DiGraph()
    for d in debates:
        nid = d["node_id"]
        g.add_node(nid, agent_id=d["agent_id"], tick=d["tick"], claim=d.get("claim", ""))
    for d in debates:
        target = d.get("target_node_id")
        if target and g.has_node(target):
            g.add_edge(d["node_id"], target, attack_type=d.get("attack_type") or "rebuttal")
    return g


def _grounded_extension(g: nx.DiGraph) -> set[str]:
    if g.number_of_nodes() == 0:
        return set()
    ge: set[str] = {n for n in g.nodes() if g.in_degree(n) == 0}
    while True:
        defeated: set[str] = set()
        for s in ge:
            defeated.update(g.successors(s))
        new_members: set[str] = set()
        for n in g.nodes():
            if n in ge or n in defeated:
                continue
            attackers = set(g.predecessors(n))
            if attackers and all(a in defeated for a in attackers):
                new_members.add(n)
        if not new_members:
            break
        ge = ge | new_members
    return ge


def _acceptance_ratio(g: nx.DiGraph) -> float:
    n = g.number_of_nodes()
    if n == 0:
        return 0.0
    return len(_grounded_extension(g)) / n


def _defeat_cycles(g: nx.DiGraph) -> int:
    return sum(1 for scc in nx.strongly_connected_components(g) if len(scc) > 1)


def _dialectical_completeness(g: nx.DiGraph) -> float:
    n = g.number_of_nodes()
    if n == 0:
        return 0.0
    ge = _grounded_extension(g)
    addressed: set[str] = set(ge)
    for s in ge:
        addressed.update(g.successors(s))
    return len(addressed) / n


def _prr_graph(debates: list[dict[str, Any]]) -> float:
    if not debates:
        return 0.0
    return sum(1 for d in debates if d.get("target_node_id")) / len(debates)


def _mean_delta_phi(debates: list[dict[str, Any]]) -> float:
    if not debates:
        return 0.0
    return sum(float(d.get("delta_phi", 0.0)) for d in debates) / len(debates)


def _acceptance_slope(debates: list[dict[str, Any]]) -> float:
    """Slope of the per-step acceptance ratio sequence built incrementally.

    For each k = 1..K, build the subgraph from the first k debates and compute
    the acceptance ratio. Then fit a least-squares slope on (k, alpha_k).
    Positive slope = acceptance grows with debate count; negative = degrades.
    """
    if len(debates) < 3:
        return 0.0

    g = nx.DiGraph()
    series: list[float] = []
    for d in debates:
        nid = d["node_id"]
        g.add_node(nid, agent_id=d["agent_id"], tick=d["tick"])
        target = d.get("target_node_id")
        if target and g.has_node(target):
            g.add_edge(nid, target)
        series.append(_acceptance_ratio(g))

    n = len(series)
    xs = list(range(1, n + 1))
    mean_x = sum(xs) / n
    mean_y = sum(series) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, series))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-seed analysis
# ---------------------------------------------------------------------------


def analyze_seed(
    hrrl_path: Path,
    langgraph_path: Path,
    seed: int,
) -> dict[str, Any]:
    """Compute full and volume-matched metrics for one seed pair."""
    logs_h = _load_logs(hrrl_path)
    logs_l = _load_logs(langgraph_path)

    debates_h_full = _debate_entries(logs_h)
    debates_l_full = _debate_entries(logs_l)

    n_h = len(debates_h_full)
    n_l = len(debates_l_full)
    k = n_l  # match HRRL volume to LangGraph

    # Primeros-K volume matching: HRRL's first K debates (early timeline).
    logs_h_trunc = _truncate_to_k_debates(logs_h, k)
    debates_h_trunc = _debate_entries(logs_h_trunc)

    # Últimos-K volume matching: HRRL's last K debates (late timeline).
    # Addresses the context-length confound raised by REV5: the first K
    # HRRL debates happen early (short context), while LangGraph's K
    # debates span the full horizon (full context). Last-K aligns HRRL's
    # window with LangGraph's timeline, so any remaining difference
    # cannot be attributed to HRRL operating on a shorter context.
    debates_h_last = _select_last_k_debates(logs_h, k)

    g_h_full = _build_subgraph(debates_h_full)
    g_h_trunc = _build_subgraph(debates_h_trunc)
    g_h_last = _build_subgraph(debates_h_last)
    g_l_full = _build_subgraph(debates_l_full)

    return {
        "seed": seed,
        "n_debates_hrrl_full": n_h,
        "n_debates_hrrl_truncated": len(debates_h_trunc),
        "n_debates_hrrl_last_k": len(debates_h_last),
        "n_debates_langgraph": n_l,
        "volume_ratio_hrrl_over_lg": round(n_h / n_l, 3) if n_l else float("inf"),
        # HRRL FULL
        "hrrl_full_acceptance": round(_acceptance_ratio(g_h_full), 4),
        "hrrl_full_defeat_cycles": _defeat_cycles(g_h_full),
        "hrrl_full_completeness": round(_dialectical_completeness(g_h_full), 4),
        "hrrl_full_prr_g": round(_prr_graph(debates_h_full), 4),
        "hrrl_full_mean_dphi": round(_mean_delta_phi(debates_h_full), 4),
        "hrrl_full_acc_slope": round(_acceptance_slope(debates_h_full), 6),
        # HRRL VOLUME-MATCHED primeros-K (first K debates, early context).
        "hrrl_vm_acceptance": round(_acceptance_ratio(g_h_trunc), 4),
        "hrrl_vm_defeat_cycles": _defeat_cycles(g_h_trunc),
        "hrrl_vm_completeness": round(_dialectical_completeness(g_h_trunc), 4),
        "hrrl_vm_prr_g": round(_prr_graph(debates_h_trunc), 4),
        "hrrl_vm_mean_dphi": round(_mean_delta_phi(debates_h_trunc), 4),
        "hrrl_vm_acc_slope": round(_acceptance_slope(debates_h_trunc), 6),
        # HRRL VOLUME-MATCHED últimos-K (last K debates, full context).
        "hrrl_vm_last_acceptance": round(_acceptance_ratio(g_h_last), 4),
        "hrrl_vm_last_defeat_cycles": _defeat_cycles(g_h_last),
        "hrrl_vm_last_completeness": round(_dialectical_completeness(g_h_last), 4),
        "hrrl_vm_last_prr_g": round(_prr_graph(debates_h_last), 4),
        "hrrl_vm_last_mean_dphi": round(_mean_delta_phi(debates_h_last), 4),
        "hrrl_vm_last_acc_slope": round(_acceptance_slope(debates_h_last), 6),
        # LangGraph FULL
        "lg_acceptance": round(_acceptance_ratio(g_l_full), 4),
        "lg_defeat_cycles": _defeat_cycles(g_l_full),
        "lg_completeness": round(_dialectical_completeness(g_l_full), 4),
        "lg_prr_g": round(_prr_graph(debates_l_full), 4),
        "lg_mean_dphi": round(_mean_delta_phi(debates_l_full), 4),
        "lg_acc_slope": round(_acceptance_slope(debates_l_full), 6),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def discover_pairs(logs_dir: Path, seeds: list[int]) -> list[tuple[int, Path, Path]]:
    pairs: list[tuple[int, Path, Path]] = []
    for s in seeds:
        h = logs_dir / f"logs_hrrl_seed{s}.json"
        l = logs_dir / f"logs_langgraph_seed{s}.json"
        if h.exists() and l.exists():
            pairs.append((s, h, l))
        else:
            logger.warning("Skipping seed %d: missing %s or %s", s, h.name, l.name)
    return pairs


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std across seeds for the key metrics, both full and volume-matched."""
    if not rows:
        return {}

    def _stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        return {
            "mean": round(statistics.fmean(values), 4),
            "std": round(statistics.stdev(values) if len(values) > 1 else 0.0, 4),
            "n": len(values),
        }

    def _col(key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    keys = [
        "hrrl_full_acceptance", "hrrl_full_defeat_cycles", "hrrl_full_completeness",
        "hrrl_full_prr_g", "hrrl_full_mean_dphi", "hrrl_full_acc_slope",
        "hrrl_vm_acceptance", "hrrl_vm_defeat_cycles", "hrrl_vm_completeness",
        "hrrl_vm_prr_g", "hrrl_vm_mean_dphi", "hrrl_vm_acc_slope",
        "hrrl_vm_last_acceptance", "hrrl_vm_last_defeat_cycles", "hrrl_vm_last_completeness",
        "hrrl_vm_last_prr_g", "hrrl_vm_last_mean_dphi", "hrrl_vm_last_acc_slope",
        "lg_acceptance", "lg_defeat_cycles", "lg_completeness",
        "lg_prr_g", "lg_mean_dphi", "lg_acc_slope",
        "n_debates_hrrl_full", "n_debates_langgraph", "volume_ratio_hrrl_over_lg",
    ]
    return {k: _stats(_col(k)) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Volume-matched analysis (Fase 0B)")
    parser.add_argument("--logs-dir", default="benchmark_results_v7")
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=[7, 17, 42, 123, 256],
        help="Seeds to process (only those with both logs present are kept).",
    )
    parser.add_argument("--csv", default="tools/volume_matched_results.csv")
    parser.add_argument("--summary", default="tools/volume_matched_summary.json")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    logs_dir = Path(args.logs_dir)
    pairs = discover_pairs(logs_dir, args.seeds)
    if not pairs:
        print("No matching log pairs found.")
        return

    rows = [analyze_seed(h, l, s) for s, h, l in pairs]

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, csv_path)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    summary = {
        "seeds_processed": [r["seed"] for r in rows],
        "metrics": summarize(rows),
    }
    sum_path = Path(args.summary)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote {sum_path}")

    # Console preview
    print("\nPer-seed (HRRL full | VM primeros-K | VM ultimos-K | LangGraph):")
    print("seed  N_h  N_vm  N_l  acc_h  acc_vm_first  acc_vm_last  acc_l  dphi_h  dphi_vm_first  dphi_vm_last  dphi_l")
    for r in rows:
        print(
            f"{r['seed']:>4}  {r['n_debates_hrrl_full']:>3}  "
            f"{r['n_debates_hrrl_truncated']:>4}  {r['n_debates_langgraph']:>3}  "
            f"{r['hrrl_full_acceptance']:>5.3f}  "
            f"{r['hrrl_vm_acceptance']:>11.3f}  "
            f"{r['hrrl_vm_last_acceptance']:>10.3f}  "
            f"{r['lg_acceptance']:>5.3f}  "
            f"{r['hrrl_full_mean_dphi']:>5.3f}  "
            f"{r['hrrl_vm_mean_dphi']:>12.3f}  "
            f"{r['hrrl_vm_last_mean_dphi']:>11.3f}  "
            f"{r['lg_mean_dphi']:>5.3f}"
        )


if __name__ == "__main__":
    main()
