"""Comparative benchmark: HRRL vs LangGraph orchestration.

Scientific question: Does endogenous homeostatic regulation (HRRL) produce richer,
more genuinely dialectical multi-agent discourse than exogenous LLM-based routing
(LangGraph), when both modes have access to identical discourse state and equal
debate-turn budgets?

Experimental design:
  1. Run HRRL for T ticks → produces N_HRRL debate turns
  2. Run LangGraph until it reaches N_HRRL debate turns (budget-matched)
  3. Compare final argument graphs using established metrics

Outcome metrics (primary):
  - AAF defeat cycles (Dung 1995): |SCC_{>1}| on attack graph
  - AAF acceptance ratio (Dung 1995): |grounded extension| / |total nodes|
  - PRR text (Marandi 2026): peer mention + stance word in debate claims
  - PRR graph (structural): fraction of debates with non-null target
  - Avg Δφ* (this paper): network-theoretic argumentative integration

Validity check (NOT a comparison metric):
  - IR (Initiative Ratio): HOMEOSTATIC turns / active turns
    IR≈1.0 for HRRL (self-initiated), IR≈0.0 for LangGraph (externally routed)

Statistical test:
  Hypotheses H1 (defeat cycles), H2 (PRR), H3 (Δφ*):
    H0: μ_HRRL ≤ μ_LG  vs  H1: μ_HRRL > μ_LG
  One-sided Welch's t-test, Bonferroni-corrected (alpha=0.05/3=0.0167).

Checkpoint/resume: after each (mode, seed) combination completes, results are
persisted to benchmark_checkpoint.json inside --output-dir. On restart, completed
combinations are skipped automatically. Use --clean to start fresh.

Usage
-----
    python benchmark.py --model gpt-4o-mini --iterations 50 --seeds 7 17 42 123 256
    python benchmark.py --modes hrrl langgraph --output-dir results
    python benchmark.py --clean   # discard checkpoint, start fresh
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from langclaw.delp_graph import ArgumentGraph
from langclaw.memory import reset_shared_store
from langclaw.metrics import (
    initiative_ratio,
    peer_reference_rate,
    peer_reference_rate_graph,
)
from langclaw.schemas import SimulationLog
from langclaw.simulation import OrchestrationMode, SotopiaEnvironment

load_dotenv()
console = Console()

DEFAULT_MODES = ["hrrl", "langgraph"]
DEFAULT_SEEDS = [7, 17, 42, 123, 256]


def _bm_checkpoint_key(mode: str, seed: int) -> str:
    return f"{mode}__seed{seed}"


def _load_bm_checkpoint(path: Path) -> dict[str, dict]:
    """Load completed benchmark runs from checkpoint file."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {r["_ck"]: r for r in data if "_ck" in r}
    except (json.JSONDecodeError, KeyError):
        return {}


def _save_bm_checkpoint(path: Path, completed: dict[str, dict]) -> None:
    """Persist all completed benchmark runs to checkpoint file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(completed.values()), f, indent=2, ensure_ascii=False)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Heuristic detection for OpenAI quota/rate-limit failures."""
    parts: list[str] = []
    cur: BaseException | None = exc
    hops = 0
    while cur is not None and hops < 8:
        parts.append(str(cur).lower())
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        hops += 1
    msg = " | ".join(parts)
    markers = [
        "rate limit",
        "insufficient_quota",
        "quota",
        "429",
        "rpd",
        "rpm",
    ]
    return any(m in msg for m in markers)

# All agent IDs for PRR text-based computation (VSM structure)
_AGENT_IDS = [
    "GOV-S1", "GOV-S2", "GOV-S3", "GOV-S4", "GOV-S5",
    "OPP-S1", "OPP-S2", "OPP-S3", "OPP-S4", "OPP-S5",
]


def _load_calibration_config(path: str | None) -> dict:
    """Load calibrated hyperparameters from calibration_results.json.

    Returns a dict with 'stimulus_weights' and 'debate_alpha' keys,
    or empty dict if no config file is provided.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        console.print(f"[yellow]Config file {path} not found, using defaults[/yellow]")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    best = data.get("best_config", {})
    console.print(
        f"[green]Loaded calibration config:[/green] "
        f"weights={best.get('weight_config_name', 'custom')}, "
        f"alpha={best.get('debate_alpha', 2.0)}"
    )
    return {
        "stimulus_weights": best.get("stimulus_weights"),
        "debate_alpha": best.get("debate_alpha", 2.0),
    }


def _run_mode(
    mode: str,
    base_url: str,
    model: str,
    api_key: str,
    iterations: int,
    seed: int | None,
    api_hard_limit: int,
    initial_deficit: float,
    stimulus_weights: dict[str, float] | None = None,
    debate_alpha: float = 2.0,
) -> tuple[list[SimulationLog], float, SotopiaEnvironment]:
    """Run simulation for the given mode. Returns (logs, elapsed_seconds, env).

    Both modes run the same number of heartbeats. The comparison is temporal
    (same time horizon), not volumetric (same debate count).

    stimulus_weights and debate_alpha are passed to all agents. For LangGraph
    mode, debate_alpha has no effect (no satiation/Q-learner), but
    stimulus_weights still affect stimulus evaluation for consistency.
    """
    env = SotopiaEnvironment(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_iterations=iterations,
        seed=seed,
        orchestration_mode=OrchestrationMode(mode),
        api_hard_limit=api_hard_limit,
        initial_deficit=initial_deficit,
        stimulus_weights=stimulus_weights,
        debate_alpha=debate_alpha,
    )
    t0 = time.perf_counter()
    logs = env.run()
    elapsed = time.perf_counter() - t0
    return logs, elapsed, env


def _build_aaf_from_logs(logs: list[SimulationLog]) -> ArgumentGraph:
    """Reconstruct a minimal ArgumentGraph for AAF metric computation.

    We rebuild the graph structure using the node IDs and target links stored
    in the simulation logs.  This avoids passing the live graph across seeds.
    """
    import uuid
    import networkx as nx
    from langclaw.delp_graph import ArgumentGraph

    g = ArgumentGraph()
    # Map from log-recorded target_node_id strings to actual node IDs in graph
    # Reconstruction: add nodes in tick order, connecting attack edges
    node_ids_seen: list[str] = []

    for log in logs:
        if log.action != "DEBATE" or not log.claim:
            continue
        # We cannot reconstruct exact node IDs from logs (they contain UUID suffixes)
        # Instead we use the graph's own node list from the last log entry
        # → use graph_nodes count as a proxy and compute metrics from live env graph
        break

    # Return empty — callers should use env.graph directly (passed as arg)
    return g


def _compute_aaf_metrics(graph: ArgumentGraph) -> dict[str, float | int]:
    """Compute AAF metrics from the live argument graph."""
    return {
        "aaf_defeat_cycles": graph.defeat_cycle_count(),
        "aaf_acceptance_ratio": round(graph.acceptance_ratio(), 4),
        "aaf_dialectical_completeness": round(graph.dialectical_completeness(), 4),
    }


def _compute_metrics(
    logs: list[SimulationLog],
    graph: ArgumentGraph,
) -> dict:
    """Derive aggregate metrics from simulation logs and the live argument graph."""
    debates = [l for l in logs if l.action == "DEBATE"]
    total_ticks = max((l.tick for l in logs), default=0)
    agents = sorted(set(l.agent_id for l in logs))

    avg_delta_phi = sum(l.delta_phi for l in debates) / len(debates) if debates else 0.0

    graph_nodes = logs[-1].graph_nodes if logs else 0
    graph_edges = logs[-1].graph_edges if logs else 0

    per_agent_debates: dict[str, int] = {}
    per_agent_avg_dphi: dict[str, float] = {}
    final_deficits: dict[str, float] = {}

    for agent_id in agents:
        ad = [l for l in debates if l.agent_id == agent_id]
        agent_logs = [l for l in logs if l.agent_id == agent_id]
        per_agent_debates[agent_id] = len(ad)
        per_agent_avg_dphi[agent_id] = (
            sum(l.delta_phi for l in ad) / len(ad) if ad else 0.0
        )
        if agent_logs:
            final_deficits[agent_id] = agent_logs[-1].deficit_after

    # Primary outcome metrics
    prr_text = peer_reference_rate(logs, _AGENT_IDS)
    prr_graph = peer_reference_rate_graph(logs)
    ir = initiative_ratio(logs)  # validity check, not comparison metric

    # AAF metrics from live graph
    aaf = _compute_aaf_metrics(graph)

    # Router call count (LangGraph overhead transparency)
    router_calls = sum(1 for l in logs if l.trigger == "ROUTER")

    # Stimulus metrics (HRRL event-driven mode)
    stim_evaluated = [l.n_stimuli_evaluated for l in logs if l.n_stimuli_evaluated > 0]
    avg_stimuli_per_tick = (
        sum(stim_evaluated) / len(stim_evaluated) if stim_evaluated else 0.0
    )
    stim_utils = [l.stimulus_utility for l in logs if l.stimulus_utility > 0]
    avg_stimulus_utility = (
        sum(stim_utils) / len(stim_utils) if stim_utils else 0.0
    )
    stimulus_driven_debates = sum(
        1 for l in debates if l.stimulus_event_id is not None
    )

    # HRRL Q-learning metrics
    rewards = [l.reward for l in logs if l.reward != 0.0]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    total_reward = sum(rewards)

    # Extract final Q-weights per agent (from last log entry per agent)
    final_q_weights: dict[str, dict] = {}
    for agent_id in agents:
        agent_logs = [l for l in logs if l.agent_id == agent_id and l.q_values]
        if agent_logs:
            final_q_weights[agent_id] = agent_logs[-1].q_values

    return {
        "total_ticks": total_ticks,
        "total_debates": len(debates),
        "avg_delta_phi": round(avg_delta_phi, 4),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        # Primary outcome metrics
        "aaf_defeat_cycles": aaf["aaf_defeat_cycles"],
        "aaf_acceptance_ratio": aaf["aaf_acceptance_ratio"],
        "aaf_dialectical_completeness": aaf["aaf_dialectical_completeness"],
        "prr_text": round(prr_text, 4),
        "prr_graph": round(prr_graph, 4),
        # Validity check (not a comparison metric)
        "ir": round(ir, 4),
        # Cost transparency
        "router_calls": router_calls,
        # Stimulus metrics
        "avg_stimuli_per_tick": round(avg_stimuli_per_tick, 2),
        "avg_stimulus_utility": round(avg_stimulus_utility, 4),
        "stimulus_driven_debates": stimulus_driven_debates,
        # HRRL Q-learning metrics
        "avg_reward": round(avg_reward, 4),
        "total_reward": round(total_reward, 4),
        "final_q_weights": final_q_weights,
        # Per-agent breakdown
        "final_deficits": {k: round(v, 4) for k, v in final_deficits.items()},
        "per_agent_debates": per_agent_debates,
        "per_agent_avg_dphi": {k: round(v, 4) for k, v in per_agent_avg_dphi.items()},
    }


def _compute_temporal_metrics(
    logs: list[SimulationLog],
    graph: ArgumentGraph,
    n_windows: int = 5,
) -> dict:
    """Compute temporal resilience metrics: windowed delta-phi, AAF, and CORE slopes.

    Divides debates into n_windows temporal windows and fits a linear
    regression slope for each metric vs window index. Negative slopes
    indicate degradation over time.
    """
    from scipy.stats import linregress as _linregress

    debates = [l for l in logs if l.action == "DEBATE" and l.claim]
    debates.sort(key=lambda l: l.tick)

    if len(debates) < n_windows:
        return {
            "window_dphi": [],
            "slope_dphi": 0.0,
            "window_acceptance": [],
            "slope_acceptance": 0.0,
            "window_core": [],
            "slope_core": 0.0,
        }

    window_size = len(debates) // n_windows

    # --- Windowed delta-phi ---
    window_dphi = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(debates)
        window = debates[start:end]
        if window:
            mean_dphi = statistics.mean(l.delta_phi for l in window)
            window_dphi.append(round(mean_dphi, 4))

    slope_dphi = 0.0
    if len(window_dphi) >= 2:
        reg = _linregress(range(len(window_dphi)), window_dphi)
        slope_dphi = round(reg.slope, 6)

    # --- Windowed AAF acceptance ratio (graph replay) ---
    window_acceptance = []
    for i in range(n_windows):
        end_idx = (i + 1) * window_size if i < n_windows - 1 else len(debates)
        partial_debates = debates[:end_idx]
        partial_graph = _replay_graph_from_debates(partial_debates)
        acc = partial_graph.acceptance_ratio()
        window_acceptance.append(round(acc, 4))

    slope_acceptance = 0.0
    if len(window_acceptance) >= 2:
        reg = _linregress(range(len(window_acceptance)), window_acceptance)
        slope_acceptance = round(reg.slope, 6)

    # --- Windowed CORE (if embeddings available) ---
    window_core: list[float] = []
    slope_core = 0.0
    try:
        from langclaw.core_metric import compute_core
        import numpy as np

        for i in range(n_windows):
            start = i * window_size
            end = start + window_size if i < n_windows - 1 else len(debates)
            window = debates[start:end]
            utterances = [l.claim or "" for l in window]

            embeddings = _get_embeddings_for_utterances(utterances)
            if embeddings is not None:
                score = compute_core(utterances, embeddings)
                window_core.append(round(score, 6))

        if len(window_core) >= 2:
            reg = _linregress(range(len(window_core)), window_core)
            slope_core = round(reg.slope, 6)
    except ImportError:
        pass

    return {
        "window_dphi": window_dphi,
        "slope_dphi": slope_dphi,
        "window_acceptance": window_acceptance,
        "slope_acceptance": slope_acceptance,
        "window_core": window_core,
        "slope_core": slope_core,
    }


def _replay_graph_from_debates(debates: list[SimulationLog]) -> ArgumentGraph:
    """Rebuild an ArgumentGraph from logged debate entries (for incremental AAF)."""
    g = ArgumentGraph()
    for d in debates:
        if d.claim:
            g.add_argument(
                agent_id=d.agent_id,
                claim=d.claim,
                target_node_id=d.target_node_id,
                attack_type=d.attack_type,
                tick=d.tick,
            )
    return g


def _get_embeddings_for_utterances(utterances: list[str]) -> "np.ndarray | None":
    """Get embeddings for utterances using OpenAI API (if available)."""
    import os
    try:
        import numpy as np
        from openai import OpenAI

        api_key = os.getenv("OPEN_AI_API_KEY", "")
        if not api_key:
            return None

        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=utterances,
        )
        embeddings = np.array([d.embedding for d in response.data], dtype=np.float64)
        return embeddings
    except Exception:
        return None


def _run_statistical_tests(hrrl_runs: list[dict], lg_runs: list[dict]) -> dict:
    """Run Welch's t-tests for temporal resilience hypotheses.

    H1 (Linguistic resilience): CORE slope HRRL > LG (one-sided)
    H2 (Quality resilience): delta-phi slope HRRL > LG (one-sided)
    H3 (Contestation resilience): acceptance-ratio slope HRRL more stable

    Bonferroni corrected alpha = 0.05 / 3 = 0.0167

    Also includes legacy aggregate tests for backward compatibility.
    """
    import math

    bonferroni_alpha = 0.05 / 3

    def welch_t(a: list[float], b: list[float], one_sided: bool = True) -> dict:
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return {"t": None, "df": None, "p": None, "cohen_d": None, "significant": None}

        mean1, mean2 = statistics.mean(a), statistics.mean(b)
        var1, var2 = statistics.variance(a), statistics.variance(b)

        se = math.sqrt(var1 / n1 + var2 / n2)
        if se == 0:
            return {"t": 0.0, "df": 0, "p": 0.5, "cohen_d": 0.0, "significant": False}

        t_stat = (mean1 - mean2) / se

        df_num = (var1 / n1 + var2 / n2) ** 2
        df_den = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = df_num / df_den if df_den > 0 else 1.0

        p_upper = _t_dist_upper_tail(abs(t_stat), df)
        if one_sided:
            p = p_upper if t_stat > 0 else 1.0 - p_upper
        else:
            p = min(1.0, 2.0 * p_upper)

        pooled_sd = math.sqrt((var1 + var2) / 2)
        cohen_d = (mean1 - mean2) / pooled_sd if pooled_sd > 0 else 0.0

        return {
            "t": round(t_stat, 4),
            "df": round(df, 2),
            "p": round(p, 4),
            "cohen_d": round(cohen_d, 4),
            "significant": p < bonferroni_alpha,
            "bonferroni_alpha": bonferroni_alpha,
        }

    # Primary hypotheses: temporal slopes
    h1_core = welch_t(
        [r.get("slope_core", 0.0) for r in hrrl_runs],
        [r.get("slope_core", 0.0) for r in lg_runs],
        one_sided=True,
    )
    h2_quality = welch_t(
        [r.get("slope_dphi", 0.0) for r in hrrl_runs],
        [r.get("slope_dphi", 0.0) for r in lg_runs],
        one_sided=True,
    )
    h3_contestation = welch_t(
        [r.get("slope_acceptance", 0.0) for r in hrrl_runs],
        [r.get("slope_acceptance", 0.0) for r in lg_runs],
        one_sided=False,
    )

    # Descriptive tests (not primary hypotheses)
    participation_equity = welch_t(
        [statistics.stdev(list(r["per_agent_debates"].values())) for r in hrrl_runs],
        [statistics.stdev(list(r["per_agent_debates"].values())) for r in lg_runs],
        one_sided=True,
    )
    acceptance_ratio = welch_t(
        [r["aaf_acceptance_ratio"] for r in hrrl_runs],
        [r["aaf_acceptance_ratio"] for r in lg_runs],
        one_sided=False,
    )

    return {
        "H1_core_slope": h1_core,
        "H2_quality_slope": h2_quality,
        "H3_contestation_slope": h3_contestation,
        "descriptive_participation": participation_equity,
        "descriptive_acceptance": acceptance_ratio,
    }


def _t_dist_upper_tail(t: float, df: float) -> float:
    """Approximate upper-tail p-value for t-distribution (one-sided).

    Uses the regularized incomplete beta function approximation.
    For small df this is an approximation; for n=5 it is adequate.
    """
    import math

    if df <= 0:
        return 0.5
    if t <= 0:
        return 0.5  # upper tail of non-positive t is >= 0.5

    # Use regularized incomplete beta: P(T > t | df) = 0.5 * I(df/(df+t^2); df/2, 0.5)
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5

    # Regularized incomplete beta via continued fraction (Lentz's method)
    try:
        p = 0.5 * _regularized_incomplete_beta(x, a, b)
    except Exception:
        p = 0.5  # fallback if numerical issues

    return max(0.0, min(1.0, p))


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction."""
    import math

    if x < 0 or x > 1:
        raise ValueError("x must be in [0, 1]")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a

    # Lentz's continued fraction
    MAX_ITER = 200
    EPS = 3e-7
    FPMIN = 1e-300

    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break

    return front * h


def _aggregate_multi_seed(runs: list[dict]) -> dict:
    """Compute mean +/- std across multiple seeded runs."""
    numeric_keys = [
        "total_debates", "avg_delta_phi", "graph_nodes", "graph_edges",
        "aaf_defeat_cycles", "aaf_acceptance_ratio", "aaf_dialectical_completeness",
        "prr_text", "prr_graph", "ir",
        "avg_stimuli_per_tick", "avg_stimulus_utility", "stimulus_driven_debates",
        "avg_reward", "total_reward",
    ]
    agg: dict[str, Any] = {}
    for key in numeric_keys:
        vals = [r[key] for r in runs if key in r]
        if vals:
            agg[key] = round(statistics.mean(vals), 4)
            agg[f"{key}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        else:
            agg[key] = 0.0
            agg[f"{key}_std"] = 0.0

    agents = sorted(set(a for r in runs for a in r.get("per_agent_debates", {})))
    agg["per_agent_debates"] = {}
    agg["per_agent_avg_dphi"] = {}
    agg["final_deficits"] = {}
    for agent_id in agents:
        vals_d = [r["per_agent_debates"].get(agent_id, 0) for r in runs]
        vals_p = [r["per_agent_avg_dphi"].get(agent_id, 0.0) for r in runs]
        vals_f = [r["final_deficits"].get(agent_id, 0.0) for r in runs]
        agg["per_agent_debates"][agent_id] = round(statistics.mean(vals_d), 1)
        agg["per_agent_avg_dphi"][agent_id] = round(statistics.mean(vals_p), 4)
        agg["final_deficits"][agent_id] = round(statistics.mean(vals_f), 4)
    return agg


def _print_comparison_table(all_metrics: dict) -> None:
    """Print a Rich comparison table to the console."""
    table = Table(
        title="Benchmark Results (mean across seeds)",
        show_lines=True,
        title_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", width=30)
    for mode in all_metrics:
        table.add_column(mode.upper(), justify="right", width=14)

    rows = [
        ("--- Setup ---", None),
        ("Total Debates", lambda m: str(m["total_debates"])),
        ("Total Ticks", lambda m: str(m.get("total_ticks", "-"))),
        ("Router Calls (overhead)", lambda m: str(m.get("router_calls", 0))),
        ("--- Outcome Metrics ---", None),
        ("AAF Defeat Cycles (H1)", lambda m: f'{m["aaf_defeat_cycles"]:.2f}'),
        ("AAF Acceptance Ratio", lambda m: f'{m["aaf_acceptance_ratio"]:.4f}'),
        ("AAF Dialectical Completeness", lambda m: f'{m["aaf_dialectical_completeness"]:.4f}'),
        ("PRR Text (H2)", lambda m: f'{m["prr_text"]:.4f}'),
        ("PRR Graph (structural)", lambda m: f'{m["prr_graph"]:.4f}'),
        ("Avg delta-phi* (H3)", lambda m: f'{m["avg_delta_phi"]:.4f}'),
        ("--- Q-Learning Metrics ---", None),
        ("Avg Reward (drive reduction)", lambda m: f'{m.get("avg_reward", 0):.4f}'),
        ("Total Reward", lambda m: f'{m.get("total_reward", 0):.4f}'),
        ("--- Stimulus Metrics ---", None),
        ("Avg Stimuli/Tick", lambda m: f'{m.get("avg_stimuli_per_tick", 0):.2f}'),
        ("Avg Stimulus Utility", lambda m: f'{m.get("avg_stimulus_utility", 0):.4f}'),
        ("Stimulus-Driven Debates", lambda m: str(int(m.get("stimulus_driven_debates", 0)))),
        ("--- Validity Check ---", None),
        ("IR (Initiative Ratio)", lambda m: f'{m["ir"]:.4f}'),
        ("Graph Nodes", lambda m: str(m["graph_nodes"])),
        ("Graph Edges", lambda m: str(m["graph_edges"])),
    ]
    for label, fn in rows:
        if fn is None:
            table.add_row(f"[bold]{label}[/bold]", *["" for _ in all_metrics])
        else:
            table.add_row(label, *(fn(all_metrics[mode]) for mode in all_metrics))

    console.print(table)


def _print_statistical_tests(tests: dict) -> None:
    """Print Welch's t-test results."""
    table = Table(
        title="Statistical Tests (Welch's t, Bonferroni alpha=0.0167)",
        show_lines=True,
        title_style="bold yellow",
    )
    table.add_column("Hypothesis", style="cyan", width=46)
    table.add_column("t", justify="right", width=8)
    table.add_column("df", justify="right", width=6)
    table.add_column("p", justify="right", width=8)
    table.add_column("Cohen d", justify="right", width=10)
    table.add_column("Significant", justify="center", width=12)

    labels = {
        "H1_core_slope": "H1: CORE slope HRRL > LG (ling. resilience)",
        "H2_quality_slope": "H2: dphi slope HRRL > LG (quality resil.)",
        "H3_contestation_slope": "H3: accept. slope HRRL vs LG (contest.)",
        "descriptive_participation": "Desc: participation equity (design)",
        "descriptive_acceptance": "Desc: acceptance ratio (2-sided)",
    }
    for key, label in labels.items():
        r = tests.get(key, {})
        if r.get("t") is None:
            table.add_row(label, "n/a", "n/a", "n/a", "n/a", "n/a")
        else:
            sig = "[green]YES[/green]" if r.get("significant") else "[red]no[/red]"
            table.add_row(
                label,
                str(r["t"]),
                str(r["df"]),
                str(r["p"]),
                str(r["cohen_d"]),
                sig,
            )

    console.print(table)
    console.print(
        "  [dim]Note: n=5 seeds; results are exploratory. "
        "Primary hypotheses test temporal slopes (resilience), not aggregates.[/dim]"
    )


def _save_comparison_charts(all_metrics: dict, all_logs: dict, output_dir: Path) -> None:
    """Generate Plotly charts comparing modes and save as HTML."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        console.print("[yellow]plotly not installed — skipping chart generation[/yellow]")
        return

    modes = list(all_metrics.keys())
    colors = ["#2ecc71", "#3498db", "#e74c3c"][:len(modes)]

    # 1. Primary outcome metrics comparison
    outcome_keys = ["aaf_defeat_cycles", "prr_text", "avg_delta_phi"]
    outcome_labels = ["AAF Defeat Cycles", "PRR (text)", "Avg delta-phi*"]
    fig_outcomes = make_subplots(
        rows=1, cols=3,
        subplot_titles=outcome_labels,
    )
    for i, (key, label) in enumerate(zip(outcome_keys, outcome_labels), start=1):
        fig_outcomes.add_trace(
            go.Bar(
                x=modes,
                y=[all_metrics[m].get(key, 0) for m in modes],
                marker_color=colors,
                text=[f'{all_metrics[m].get(key, 0):.3f}' for m in modes],
                textposition="auto",
                showlegend=False,
            ),
            row=1, col=i,
        )
    fig_outcomes.update_layout(
        title="Primary Outcome Metrics: HRRL vs LangGraph",
        template="plotly_dark",
    )
    fig_outcomes.write_html(str(output_dir / "outcome_metrics_comparison.html"))

    # 2. Deficit evolution per agent per mode
    fig_deficit = make_subplots(
        rows=1, cols=len(modes),
        subplot_titles=[f"Mode: {m}" for m in modes],
        shared_yaxes=True,
    )
    agent_colors = {
        "GOV-S1": "#2ecc71", "GOV-S2": "#27ae60", "GOV-S3": "#1abc9c",
        "GOV-S4": "#16a085", "GOV-S5": "#0e6655",
        "OPP-S1": "#e74c3c", "OPP-S2": "#c0392b", "OPP-S3": "#e67e22",
        "OPP-S4": "#d35400", "OPP-S5": "#a93226",
    }
    for col_idx, mode in enumerate(modes, start=1):
        logs = all_logs.get(mode, [])
        agents = sorted(set(l.agent_id for l in logs))
        for agent_id in agents:
            agent_logs = [l for l in logs if l.agent_id == agent_id]
            fig_deficit.add_trace(
                go.Scatter(
                    x=[l.tick for l in agent_logs],
                    y=[l.deficit_after for l in agent_logs],
                    mode="lines",
                    name=agent_id if col_idx == 1 else None,
                    line=dict(color=agent_colors.get(agent_id, "#95a5a6")),
                    showlegend=(col_idx == 1),
                ),
                row=1, col=col_idx,
            )
    fig_deficit.update_layout(
        title="Epistemic Deficit Evolution by Orchestration Mode",
        template="plotly_dark",
        height=400,
    )
    fig_deficit.write_html(str(output_dir / "deficit_evolution.html"))

    console.print(f"  [green]Charts saved to {output_dir}/[/green]")


def main() -> None:
    default_api_key = os.getenv("OPEN_AI_API_KEY", "ollama")
    default_base_url = (
        "https://api.openai.com/v1" if default_api_key != "ollama"
        else "http://localhost:11434/v1"
    )
    default_model = "gpt-5-nano" if default_api_key != "ollama" else "llama3"

    parser = argparse.ArgumentParser(
        description="LangClaw Benchmark -- HRRL vs LangGraph"
    )
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--api-key", default=default_api_key)
    parser.add_argument(
        "--iterations", type=int, default=80,
        help="Max ticks for HRRL. LangGraph runs until matching debate count.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Seeds for multi-run statistical analysis. Default: 7 17 42 123 256",
    )
    parser.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES,
        choices=["hrrl", "langgraph", "round-robin", "random"],
        help="Orchestration modes to benchmark.",
    )
    parser.add_argument("--api-hard-limit", type=int, default=500)
    parser.add_argument("--initial-deficit", type=float, default=0.5)
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument(
        "--config", default=None,
        help="Path to calibration_results.json from calibrate_hyperparams.py",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Discard checkpoint and start fresh.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "benchmark_checkpoint.json"

    if args.clean and checkpoint_path.exists():
        checkpoint_path.unlink()
        console.print("[yellow]Checkpoint cleared — starting fresh[/yellow]")

    completed = _load_bm_checkpoint(checkpoint_path)
    if completed:
        console.print(
            f"[green]Resuming: {len(completed)} seed/mode combos already completed, "
            f"skipping them.[/green]"
        )

    cal = _load_calibration_config(args.config)
    cal_stimulus_weights = cal.get("stimulus_weights")
    cal_debate_alpha = cal.get("debate_alpha", 2.0)

    # Per-mode run accumulators (reload from checkpoint)
    mode_runs: dict[str, list[dict]] = {m: [] for m in args.modes}
    last_logs: dict[str, list[SimulationLog]] = {}

    for ck_entry in completed.values():
        m = ck_entry.get("_mode")
        if m and m in mode_runs:
            metrics = {k: v for k, v in ck_entry.items() if not k.startswith("_")}
            mode_runs[m].append(metrics)

    for seed in args.seeds:
        console.rule(f"[bold yellow]Seed: {seed}[/bold yellow]")

        for mode in args.modes:
            ck_key = _bm_checkpoint_key(mode, seed)

            if ck_key in completed:
                console.print(
                    f"  [dim]{mode.upper()} seed={seed} — already done, skipping[/dim]"
                )
                continue

            console.rule(
                f"[bold cyan]Running {mode.upper()} "
                f"(T={args.iterations}, seed={seed})[/bold cyan]"
            )

            try:
                logs, elapsed, env = _run_mode(
                    mode=mode,
                    base_url=args.base_url,
                    model=args.model,
                    api_key=args.api_key,
                    iterations=args.iterations,
                    seed=seed,
                    api_hard_limit=args.api_hard_limit,
                    initial_deficit=args.initial_deficit,
                    stimulus_weights=cal_stimulus_weights,
                    debate_alpha=cal_debate_alpha,
                )
            except Exception as exc:
                _save_bm_checkpoint(checkpoint_path, completed)
                if _is_rate_limit_error(exc):
                    console.print(
                        f"[yellow]Paused due to API rate/quota limit at mode={mode}, "
                        f"seed={seed}.[/yellow] Checkpoint saved. Re-run the same "
                        "command to resume."
                    )
                    raise SystemExit(75) from exc
                raise

            metrics = _compute_metrics(logs, env.graph)
            temporal = _compute_temporal_metrics(logs, env.graph, n_windows=5)
            metrics.update(temporal)
            n_debates = metrics["total_debates"]

            if mode == "langgraph" and env._router is not None:
                console.print(
                    f"  {mode.upper()}: {n_debates} debates, "
                    f"{env._router.router_call_count} router LLM calls, "
                    f"{elapsed:.1f}s"
                )
            else:
                console.print(f"  {mode.upper()}: {n_debates} debates in {elapsed:.1f}s")

            mode_runs[mode].append(metrics)
            last_logs[mode] = logs

            # Save per-seed logs
            safe_mode = mode.replace("-", "_")
            log_path = output_dir / f"logs_{safe_mode}_seed{seed}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([e.model_dump() for e in logs], f, indent=2, ensure_ascii=False)

            # Checkpoint after each (mode, seed) completes
            ck_entry = {"_ck": ck_key, "_mode": mode, "_seed": seed, **metrics}
            completed[ck_key] = ck_entry
            _save_bm_checkpoint(checkpoint_path, completed)

    # Aggregate across seeds
    agg_all = {mode: _aggregate_multi_seed(runs) for mode, runs in mode_runs.items()}

    console.rule("[bold magenta]Benchmark Results[/bold magenta]")
    _print_comparison_table(agg_all)

    # Statistical tests (HRRL vs LangGraph only)
    if "hrrl" in mode_runs and "langgraph" in mode_runs:
        console.rule("[bold yellow]Statistical Tests[/bold yellow]")
        tests = _run_statistical_tests(mode_runs["hrrl"], mode_runs["langgraph"])
        _print_statistical_tests(tests)
    else:
        tests = {}

    # Per-metric std
    console.print(f"\n[bold]Seeds used:[/bold] {args.seeds}")
    for mode in args.modes:
        debates = [r["total_debates"] for r in mode_runs[mode]]
        console.print(f"[bold]{mode.upper()} debates per seed:[/bold] {debates}")
    console.print("\n[bold]Std dev across seeds:[/bold]")
    for mode in args.modes:
        agg = agg_all[mode]
        for key in ["aaf_defeat_cycles", "prr_text", "avg_delta_phi"]:
            console.print(f"  {mode} {key}: {agg.get(f'{key}_std', 0):.4f}")

    # Save report (redact API keys)
    safe_config = dict(vars(args))
    if "api_key" in safe_config:
        safe_config["api_key"] = "***REDACTED***"
    report = {
        "aggregate": agg_all,
        "per_seed": mode_runs,
        "statistical_tests": tests,
        "config": safe_config,
        "calibration": cal if cal else {"note": "defaults (no calibration file)"},
    }
    with open(output_dir / "benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _save_comparison_charts(agg_all, last_logs, output_dir)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        console.print(f"[dim]Checkpoint {checkpoint_path} cleaned up[/dim]")

    console.print(f"\n[green]All results saved to {output_dir}/[/green]")


if __name__ == "__main__":
    main()
