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
  - PRR text (content engagement): token overlap between claim and target claim
  - PRR graph (structural): fraction of debates with non-null target
  - Avg g (quality signal): engagement × novelty + diversity (A3 gate)

Validity check (NOT a comparison metric):
  - IR (Initiative Ratio): HOMEOSTATIC turns / active turns
    IR≈1.0 for HRRL (self-initiated), IR≈0.0 for LangGraph (externally routed)

Statistical test:
  Hypotheses H1 (defeat cycles), H2 (PRR), H3 (quality signal g):
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
from langclaw.seeds import SeedFactory
from langclaw.simulation import OrchestrationMode, SotopiaEnvironment

load_dotenv()
console = Console()
logger = logging.getLogger(__name__)

DEFAULT_MODES = ["epr", "epr_q", "epr_sham", "langgraph"]
EXPERIMENT_MASTER_SEED = 20260308
DEFAULT_SEEDS = SeedFactory.derive_experiment_seeds(EXPERIMENT_MASTER_SEED, n=20)
JUDGE_SEED = SeedFactory(EXPERIMENT_MASTER_SEED).get("judge_llm")


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
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(list(completed.values()), f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _run_checkpoint_path(output_dir: Path, mode: str, seed: int | None) -> Path:
    safe_seed = "none" if seed is None else str(seed)
    return output_dir / "run_checkpoints" / f"{mode}__seed{safe_seed}.json"


def _load_run_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _save_run_checkpoint(
    path: Path,
    *,
    mode: str,
    seed: int | None,
    iterations: int,
    next_tick: int,
    env: SotopiaEnvironment,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "seed": seed,
        "iterations": iterations,
        "next_tick": next_tick,
        "env": env.to_checkpoint(),
    }
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


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
        "lambda_rate": best.get("lambda_rate", best.get("lambda", 0.05)),
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
    lambda_rate: float = 0.05,
    run_checkpoint_path: Path | None = None,
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_seed: int | None = None,
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
        lambda_rate=lambda_rate,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_seed=judge_seed,
    )
    start_tick = 1
    if run_checkpoint_path is not None:
        checkpoint_payload = _load_run_checkpoint(run_checkpoint_path)
        if checkpoint_payload and checkpoint_payload.get("mode") == mode and checkpoint_payload.get("seed") == seed:
            env.load_checkpoint(checkpoint_payload.get("env", {}))
            start_tick = int(checkpoint_payload.get("next_tick", 1))

    t0 = time.perf_counter()
    for tick in range(start_tick, iterations + 1):
        env.run_single_tick(tick)
        if run_checkpoint_path is not None:
            _save_run_checkpoint(
                run_checkpoint_path,
                mode=mode,
                seed=seed,
                iterations=iterations,
                next_tick=tick + 1,
                env=env,
            )
    elapsed = time.perf_counter() - t0
    if run_checkpoint_path is not None and run_checkpoint_path.exists():
        run_checkpoint_path.unlink()
    return env.logs, elapsed, env


def _build_aaf_from_logs(logs: list[SimulationLog]) -> ArgumentGraph:
    """Reconstruct a minimal ArgumentGraph for AAF metric computation.

    We rebuild the graph structure using the node IDs and target links stored
    in the simulation logs.  This avoids passing the live graph across seeds.
    """
    g = ArgumentGraph()
    for log in logs:
        if log.action != "DEBATE" or not log.claim:
            continue
        g.add_argument(
            agent_id=log.agent_id,
            claim=log.claim,
            target_node_id=log.target_node_id,
            attack_type=log.attack_type,
            tick=log.tick,
            node_id=log.node_id,
        )
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
        if d.claim and d.node_id:
            g.add_argument(
                agent_id=d.agent_id,
                claim=d.claim,
                target_node_id=d.target_node_id,
                attack_type=d.attack_type,
                tick=d.tick,
                node_id=d.node_id,
            )
    return g


def _get_embeddings_for_utterances(utterances: list[str]) -> "np.ndarray | None":
    """Get embeddings for utterances for the CORE metric.

    Uses OpenAI text-embedding-3-small: cheaper (~$0.02/1M tokens vs
    ~$0.069/1M for GLM embedding-3) and stronger multilingual quality
    for the Spanish debates. Requires OPEN_AI_API_KEY. Embeddings are
    used only for the offline CORE diagnostic, not for the experiment
    itself (which runs entirely on DeepSeek + GLM).
    Returns an (N, D) array or None if unavailable.
    """
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


def _detect_red_flags(
    mode: str,
    seed: int,
    metrics: dict[str, Any],
    logs: list[SimulationLog],
) -> dict[str, Any]:
    """Detect severe validity/runtime anomalies after a completed run.

    These checks are intentionally conservative: they target patterns that are
    hard to reconcile with a healthy run and therefore justify stopping the
    benchmark before spending more budget.
    """
    alerts: list[dict[str, Any]] = []
    debates = [l for l in logs if l.action == "DEBATE" and l.claim]
    q_magnitudes = [
        abs(float(v))
        for log in logs
        for v in (log.q_values or {}).values()
    ]
    max_abs_q = max(q_magnitudes, default=0.0)
    final_deficits = list((metrics.get("final_deficits") or {}).values())
    median_final_deficit = statistics.median(final_deficits) if final_deficits else 0.0
    window_acceptance = metrics.get("window_acceptance") or []
    final_acceptance = float(metrics.get("aaf_acceptance_ratio", 0.0))
    avg_reward = float(metrics.get("avg_reward", 0.0))

    if (
        len(window_acceptance) >= 3
        and all(abs(float(v) - 1.0) < 1e-9 for v in window_acceptance)
        and final_acceptance < 0.95
        and len(debates) >= 20
    ):
        alerts.append({
            "severity": "critical",
            "code": "acceptance_replay_mismatch",
            "message": (
                "Windowed acceptance stayed at 1.0 while final acceptance was materially lower. "
                "This usually indicates graph replay or node-link reconstruction failure."
            ),
            "evidence": {
                "window_acceptance": window_acceptance,
                "final_acceptance_ratio": final_acceptance,
                "debates": len(debates),
            },
        })

    if max_abs_q > 1_000.0:
        alerts.append({
            "severity": "critical",
            "code": "q_value_explosion",
            "message": (
                "Observed Q-values far outside the expected scale of homeostatic rewards, "
                "suggesting numerical instability or learner divergence."
            ),
            "evidence": {
                "max_abs_q_value": round(max_abs_q, 4),
            },
        })

    if mode == "hrrl" and avg_reward < -0.25 and median_final_deficit > 2.0:
        alerts.append({
            "severity": "critical",
            "code": "negative_homeostatic_drift",
            "message": (
                "Average reward was strongly negative while final deficits remained high, "
                "which suggests the controller is increasing drive rather than regulating it."
            ),
            "evidence": {
                "avg_reward": round(avg_reward, 4),
                "median_final_deficit": round(median_final_deficit, 4),
            },
        })

    if mode == "hrrl" and metrics.get("ir") not in (None, 1.0):
        alerts.append({
            "severity": "warning",
            "code": "unexpected_ir",
            "message": "HRRL initiative ratio deviated from the expected endogenous value.",
            "evidence": {
                "initiative_ratio": metrics.get("ir"),
            },
        })

    recent_tail = []
    for log in logs[-12:]:
        recent_tail.append({
            "tick": log.tick,
            "agent_id": log.agent_id,
            "action": log.action,
            "reward": log.reward,
            "delta_phi": log.delta_phi,
            "deficit_after": log.deficit_after,
            "q_values": log.q_values,
        })

    severity_rank = {"warning": 1, "critical": 2}
    overall = "ok"
    if alerts:
        overall = max(alerts, key=lambda a: severity_rank.get(a["severity"], 0))["severity"]

    return {
        "mode": mode,
        "seed": seed,
        "status": overall,
        "alerts": alerts,
        "summary": {
            "total_debates": metrics.get("total_debates", 0),
            "aaf_acceptance_ratio": final_acceptance,
            "avg_reward": avg_reward,
            "max_abs_q_value": round(max_abs_q, 4),
            "median_final_deficit": round(median_final_deficit, 4),
        },
        "recent_log_tail": recent_tail,
    }


def _explain_red_flags_with_llm(
    health_report: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> dict[str, Any] | None:
    """Use an LLM to produce a concise explanation for red-flagged runs."""
    alerts = health_report.get("alerts") or []
    if not alerts:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        prompt = (
            "You are diagnosing an experimental run in a multi-agent benchmark.\n"
            "Explain briefly what likely went wrong, why it threatens validity, and what to inspect next.\n"
            "Be concrete and skeptical. Do not speculate beyond the evidence.\n\n"
            f"Health report:\n{json.dumps(health_report, ensure_ascii=False, indent=2)}\n\n"
            "Return JSON with keys: probable_cause, validity_risk, next_checks."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1200,
            temperature=0.0,
            seed=JUDGE_SEED,
        )
        raw = (response.choices[0].message.content or "").strip()
        return {
            "model": model,
            "raw": raw,
        }
    except Exception as exc:
        return {
            "error": f"LLM explanation failed: {exc}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Fix 4: Wilcoxon signed-rank paired test (primary statistical test)
# Fix 7: Dual LLM judges + Cohen κ
# Fix 8: Demote g slope to diagnostic; H1 = judge quality
# Fix 9: StimulusEvaluator weight sensitivity profiles
# ──────────────────────────────────────────────────────────────────────────────

_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "uniform": {
        "w_faction": 0.20, "w_centrality": 0.20, "w_memory": 0.20,
        "w_novelty": 0.20, "w_pressure": 0.20,
    },
    "faction-heavy": {
        "w_faction": 0.50, "w_centrality": 0.125, "w_memory": 0.125,
        "w_novelty": 0.125, "w_pressure": 0.125,
    },
    "centrality-heavy": {
        "w_faction": 0.125, "w_centrality": 0.50, "w_memory": 0.125,
        "w_novelty": 0.125, "w_pressure": 0.125,
    },
    "memory-heavy": {
        "w_faction": 0.125, "w_centrality": 0.125, "w_memory": 0.50,
        "w_novelty": 0.125, "w_pressure": 0.125,
    },
    "pressure-heavy": {
        "w_faction": 0.125, "w_centrality": 0.125, "w_memory": 0.125,
        "w_novelty": 0.125, "w_pressure": 0.50,
    },
}


def get_weight_profile(name: str) -> dict[str, float]:
    """Return the StimulusEvaluator weight dict for a named profile."""
    return dict(_WEIGHT_PROFILES.get(name, _WEIGHT_PROFILES["uniform"]))


def _wilcoxon_paired(
    condition_runs: list[dict],
    baseline_runs: list[dict],
    metric_key: str,
    one_sided: bool = True,
) -> dict:
    """Wilcoxon signed-rank test on paired (by seed) samples.

    Non-parametric: does not assume normality. Preferred over Welch's t
    for small N and unknown distributions.

    Returns dict with W statistic, p-value, effect size r = W/√n, and
    significance flag at Bonferroni-corrected alpha.
    """
    from scipy.stats import wilcoxon

    cond_by_seed = {r["_seed"]: r.get(metric_key, 0.0) for r in condition_runs if "_seed" in r}
    base_by_seed = {r["_seed"]: r.get(metric_key, 0.0) for r in baseline_runs if "_seed" in r}

    common_seeds = sorted(set(cond_by_seed) & set(base_by_seed))
    if len(common_seeds) < 5:
        return {"W": None, "p": None, "r": None, "significant": None,
                "n_pairs": len(common_seeds), "note": "Too few paired samples (need ≥5)"}

    diffs = [cond_by_seed[s] - base_by_seed[s] for s in common_seeds]
    non_zero = [d for d in diffs if d != 0]
    if len(non_zero) < 5:
        return {"W": None, "p": None, "r": None, "significant": None,
                "n_pairs": len(common_seeds), "note": "Too few non-zero differences"}

    try:
        result = wilcoxon(
            [cond_by_seed[s] for s in common_seeds],
            [base_by_seed[s] for s in common_seeds],
            alternative="greater" if one_sided else "two-sided",
        )
        w_stat = result.statistic
        p_val = result.pvalue
    except Exception as exc:
        return {"W": None, "p": None, "r": None, "significant": None,
                "n_pairs": len(common_seeds), "note": f"Wilcoxon failed: {exc}"}

    n = len(non_zero)
    r_effect = w_stat / (n ** 0.5) if n > 0 else 0.0

    bonferroni_alpha = 0.05 / 3
    return {
        "W": round(w_stat, 4),
        "p": round(p_val, 6),
        "r": round(r_effect, 4),
        "significant": p_val < bonferroni_alpha,
        "bonferroni_alpha": bonferroni_alpha,
        "n_pairs": len(common_seeds),
    }


def _cohen_kappa(scores1: list, scores2: list) -> dict:
    """Compute Cohen's κ for inter-rater agreement.

    Accepts continuous scores (bins into quartiles) or categorical labels.
    Returns κ, observed agreement, and agreement category.
    """
    if len(scores1) != len(scores2) or len(scores1) < 2:
        return {"kappa": None, "p0": None, "note": "Insufficient data"}

    try:
        import numpy as np
        from sklearn.metrics import cohen_kappa_score

        s1 = np.array(scores1, dtype=float)
        s2 = np.array(scores2, dtype=float)

        if len(set(scores1)) > 5 or len(set(scores2)) > 5:
            combined = np.concatenate([s1, s2])
            q33, q66 = np.percentile(combined, [33.3, 66.7])
            def bin(v):
                if v <= q33: return "low"
                elif v <= q66: return "mid"
                else: return "high"
            b1 = [bin(v) for v in s1]
            b2 = [bin(v) for v in s2]
        else:
            b1 = [str(round(v)) for v in s1]
            b2 = [str(round(v)) for v in s2]

        kappa = cohen_kappa_score(b1, b2)
        p0 = sum(a == b for a, b in zip(b1, b2)) / len(b1)

        if kappa < 0.20:
            category = "slight"
        elif kappa < 0.40:
            category = "fair"
        elif kappa < 0.60:
            category = "moderate"
        elif kappa < 0.80:
            category = "substantial"
        else:
            category = "almost perfect"

        return {
            "kappa": round(kappa, 4),
            "p0": round(p0, 4),
            "category": category,
            "n_items": len(b1),
        }
    except ImportError:
        return {"kappa": None, "p0": None, "note": "scikit-learn not installed"}
    except Exception as exc:
        return {"kappa": None, "p0": None, "note": f"κ computation failed: {exc}"}


_JUDGE_PROMPT = """\
You are an expert judge evaluating the quality of a debate argument in context.

## Debate Topic
{debate_topic}

## Argument Under Evaluation
Claim: {claim}
Author: {agent_id}
Attack type: {attack_type}
Tick: {tick}

## Target Being Attacked
{target_context}

## Recent Debate Context (last 5 arguments)
{recent_context}

Evaluate this argument on a 1-10 scale considering:
1. Logical rigor — is the reasoning sound and well-structured?
2. Engagement — does it directly address the specific content of the target argument it attacks? (Not just the general topic.)
3. Persuasive force — would this advance the debate meaningfully, or is it a repetition/generic statement?
4. Originality — does it introduce a novel angle, evidence, or line of reasoning not already present in the debate?

Output ONLY a JSON object: {{"score": <integer 1-10>, "rationale": "<one sentence>"}}
"""


def _run_llm_judge(
    logs: list[SimulationLog],
    *,
    model: str,
    base_url: str,
    api_key: str,
) -> list[dict]:
    """Run an offline LLM judge over all debate claims.

    Returns a list of {node_id, agent_id, claim, score, rationale} dicts.
    """
    from openai import OpenAI
    from langclaw.agent import DEBATE_TOPIC

    debates = [l for l in logs if l.action == "DEBATE" and l.claim]
    if not debates:
        return []

    # Build node_id -> claim lookup for target context
    node_claims: dict[str, str] = {}
    node_agents: dict[str, str] = {}
    for l in logs:
        if l.node_id and l.claim:
            node_claims[l.node_id] = l.claim
            node_agents[l.node_id] = l.agent_id

    # Build ordered list of debate turns for recent context
    debate_sequence = list(debates)

    client = OpenAI(base_url=base_url, api_key=api_key)
    results: list[dict] = []

    for idx, d in enumerate(debates):
        # Build target context
        if d.target_node_id and d.target_node_id in node_claims:
            target_claim = node_claims[d.target_node_id]
            target_agent = node_agents.get(d.target_node_id, "unknown")
            target_context = (
                f"Node ID: {d.target_node_id}\n"
                f"Author: {target_agent}\n"
                f"Claim: \"{target_claim}\""
            )
        else:
            target_context = "None (root argument — no target being attacked)"

        # Build recent context (up to 5 preceding debate turns)
        recent_start = max(0, idx - 5)
        recent_turns = debate_sequence[recent_start:idx]
        if recent_turns:
            recent_lines = []
            for rt in recent_turns:
                rt_target = rt.target_node_id or "root"
                recent_lines.append(
                    f"[{rt.agent_id}] (tick {rt.tick}) -> {rt_target}: \"{rt.claim[:200]}\""
                )
            recent_context = "\n".join(recent_lines)
        else:
            recent_context = "No prior arguments."

        prompt = _JUDGE_PROMPT.format(
            debate_topic=DEBATE_TOPIC,
            claim=d.claim,
            agent_id=d.agent_id,
            attack_type=d.attack_type or "none",
            tick=d.tick,
            target_context=target_context,
            recent_context=recent_context,
        )
        try:
            judge_extra: dict = {"temperature": 0.0, "seed": JUDGE_SEED}
            judge_body: dict = {}
            if "glm" in model.lower():
                judge_body["do_sample"] = False
            kwargs: dict = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 300,
                **judge_extra,
            }
            if judge_body:
                kwargs["extra_body"] = judge_body
            response = client.chat.completions.create(**kwargs)
            raw = (response.choices[0].message.content or "").strip()
            import re
            json_match = re.search(r'\{[^}]+\}', raw)
            if json_match:
                parsed = json.loads(json_match.group())
                score = int(parsed.get("score", 0))
                score = max(1, min(10, score))
            else:
                score = 0
            results.append({
                "node_id": d.node_id,
                "agent_id": d.agent_id,
                "claim": d.claim,
                "score": score,
                "rationale": parsed.get("rationale", "") if json_match else raw[:200],
            })
        except Exception as exc:
            logger.warning("Judge %s failed on node %s: %s", model, d.node_id, exc)
            results.append({
                "node_id": d.node_id,
                "agent_id": d.agent_id,
                "claim": d.claim,
                "score": 0,
                "rationale": f"judge_error: {exc}",
            })

    return results


def _run_judges_for_all_modes(
    all_logs: dict[str, list[SimulationLog]],
    judge_configs: list[dict],
) -> dict:
    """Run multiple LLM judges over all modes' logs.

    judge_configs: list of {model, base_url, api_key} dicts.

    Returns:
        {
            mode: {
                judge_name: [score_dicts],
                ...
                "kappa": {...},
                "avg_judge_score": float,
            }
        }
    """
    results: dict[str, Any] = {}

    for mode, logs in all_logs.items():
        mode_results: dict[str, Any] = {}
        all_scores: list[list[float]] = []

        for jc in judge_configs:
            judge_name = jc["model"]
            console.print(f"  [dim]Judging {mode} with {judge_name}...[/dim]")
            scores = _run_llm_judge(
                logs,
                model=jc["model"],
                base_url=jc["base_url"],
                api_key=jc["api_key"],
            )
            mode_results[judge_name] = scores
            all_scores.append([s["score"] for s in scores])

        if len(all_scores) >= 2:
            min_len = min(len(s) for s in all_scores)
            mode_results["kappa"] = _cohen_kappa(
                all_scores[0][:min_len], all_scores[1][:min_len]
            )
            combined = [
                (all_scores[0][i] + all_scores[1][i]) / 2
                for i in range(min_len)
                if all_scores[0][i] > 0 and all_scores[1][i] > 0
            ]
            mode_results["avg_judge_score"] = (
                round(statistics.mean(combined), 4) if combined else 0.0
            )
        elif len(all_scores) == 1 and all_scores[0]:
            mode_results["avg_judge_score"] = round(statistics.mean(all_scores[0]), 4)
        else:
            mode_results["avg_judge_score"] = 0.0

        results[mode] = mode_results

    return results


def _run_statistical_tests(
    condition_runs: list[dict],
    baseline_runs: list[dict],
) -> dict:
    """Run Wilcoxon signed-rank paired tests (primary) and Welch's t (secondary).

    H1 (Judge quality — PRIMARY): avg_judge_score EPR > LangGraph (one-sided)
    H2 (Peer reference): PRR_text EPR > LangGraph (one-sided)
    H3 (Dialectical structure): defeat_cycles EPR > LangGraph (one-sided)

    Diagnostic (not primary):
    - g slope: temporal quality trend (demoted from H3 to diagnostic)
    - CORE slope: linguistic resilience
    - Acceptance slope: contestation stability

    Bonferroni corrected alpha = 0.05 / 3 = 0.0167
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

    # Primary hypotheses: Wilcoxon signed-rank paired tests
    h1_judge = _wilcoxon_paired(
        condition_runs, baseline_runs, "avg_judge_score", one_sided=True,
    )
    h2_prr = _wilcoxon_paired(
        condition_runs, baseline_runs, "prr_text", one_sided=True,
    )
    h3_cycles = _wilcoxon_paired(
        condition_runs, baseline_runs, "aaf_defeat_cycles", one_sided=True,
    )

    # Diagnostic: temporal slopes (demoted from primary)
    diag_dphi_slope = _wilcoxon_paired(
        condition_runs, baseline_runs, "slope_dphi", one_sided=True,
    )
    diag_core_slope = _wilcoxon_paired(
        condition_runs, baseline_runs, "slope_core", one_sided=True,
    )
    diag_acceptance_slope = _wilcoxon_paired(
        condition_runs, baseline_runs, "slope_acceptance", one_sided=False,
    )

    # Secondary: Welch's t on aggregates (for comparison with Wilcoxon)
    h1_judge_welch = welch_t(
        [r.get("avg_judge_score", 0.0) for r in condition_runs],
        [r.get("avg_judge_score", 0.0) for r in baseline_runs],
        one_sided=True,
    )
    h2_prr_welch = welch_t(
        [r.get("prr_text", 0.0) for r in condition_runs],
        [r.get("prr_text", 0.0) for r in baseline_runs],
        one_sided=True,
    )
    h3_cycles_welch = welch_t(
        [r.get("aaf_defeat_cycles", 0.0) for r in condition_runs],
        [r.get("aaf_defeat_cycles", 0.0) for r in baseline_runs],
        one_sided=True,
    )

    # Descriptive tests (not primary hypotheses)
    participation_equity = welch_t(
        [statistics.stdev(list(r["per_agent_debates"].values())) for r in condition_runs],
        [statistics.stdev(list(r["per_agent_debates"].values())) for r in baseline_runs],
        one_sided=True,
    )
    acceptance_ratio = welch_t(
        [r["aaf_acceptance_ratio"] for r in condition_runs],
        [r["aaf_acceptance_ratio"] for r in baseline_runs],
        one_sided=False,
    )

    return {
        # Primary: Wilcoxon signed-rank paired
        "H1_judge_quality": h1_judge,
        "H2_prr_text": h2_prr,
        "H3_defeat_cycles": h3_cycles,
        # Diagnostic (demoted from primary)
        "diag_dphi_slope": diag_dphi_slope,
        "diag_core_slope": diag_core_slope,
        "diag_acceptance_slope": diag_acceptance_slope,
        # Secondary: Welch's t for comparison
        "H1_judge_quality_welch": h1_judge_welch,
        "H2_prr_text_welch": h2_prr_welch,
        "H3_defeat_cycles_welch": h3_cycles_welch,
        # Descriptive
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
        ("Avg g (quality signal)", lambda m: f'{m["avg_delta_phi"]:.4f}'),
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
    """Print Wilcoxon and Welch's t-test results."""
    table = Table(
        title="Statistical Tests (Wilcoxon signed-rank primary, Bonferroni alpha=0.0167)",
        show_lines=True,
        title_style="bold yellow",
    )
    table.add_column("Hypothesis", style="cyan", width=50)
    table.add_column("Test", justify="right", width=8)
    table.add_column("W/t", justify="right", width=8)
    table.add_column("p", justify="right", width=8)
    table.add_column("r/d", justify="right", width=8)
    table.add_column("Sig.", justify="center", width=6)

    labels = {
        "H1_judge_quality": ("H1: Judge quality EPR > LG (PRIMARY)", "Wilcoxon"),
        "H2_prr_text": ("H2: PRR text EPR > LG", "Wilcoxon"),
        "H3_defeat_cycles": ("H3: Defeat cycles EPR > LG", "Wilcoxon"),
        "diag_dphi_slope": ("Diag: g slope (demoted from H3)", "Wilcoxon"),
        "diag_core_slope": ("Diag: CORE slope", "Wilcoxon"),
        "diag_acceptance_slope": ("Diag: acceptance slope", "Wilcoxon"),
        "H1_judge_quality_welch": ("H1: Judge quality (Welch check)", "Welch"),
        "H2_prr_text_welch": ("H2: PRR text (Welch check)", "Welch"),
        "H3_defeat_cycles_welch": ("H3: Defeat cycles (Welch check)", "Welch"),
        "descriptive_participation": ("Desc: participation equity", "Welch"),
        "descriptive_acceptance": ("Desc: acceptance ratio (2-sided)", "Welch"),
    }
    for key, (label, test_type) in labels.items():
        r = tests.get(key, {})
        if r.get("W") is not None:
            sig = "[green]YES[/green]" if r.get("significant") else "[red]no[/red]"
            table.add_row(label, test_type, str(r["W"]), str(r["p"]), str(r.get("r", "-")), sig)
        elif r.get("t") is not None:
            sig = "[green]YES[/green]" if r.get("significant") else "[red]no[/red]"
            table.add_row(label, test_type, str(r["t"]), str(r["p"]), str(r.get("cohen_d", "-")), sig)
        else:
            table.add_row(label, test_type, "n/a", "n/a", "n/a", "n/a")

    console.print(table)
    console.print(
        "  [dim]Note: Wilcoxon signed-rank is the primary test (non-parametric, paired by seed). "
        "Welch's t shown as secondary check. H1 = judge quality (Fix 8: Δφ* demoted to diagnostic).[/dim]"
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
    outcome_labels = ["AAF Defeat Cycles", "PRR (text)", "Avg quality signal g"]
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


def _run_preflight(args: argparse.Namespace, cal: dict[str, Any]) -> int:
    """Run a short safety check before the full benchmark."""
    cal_stimulus_weights = cal.get("stimulus_weights")
    cal_debate_alpha = cal.get("debate_alpha", 2.0)
    cal_lambda_rate = cal.get("lambda_rate", 0.05)
    preflight_dir = Path(args.output_dir) / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seeds[0]
    reports: list[dict[str, Any]] = []

    for mode in args.modes:
        logs, elapsed, env = _run_mode(
            mode=mode,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            iterations=args.preflight_ticks,
            seed=seed,
            api_hard_limit=args.api_hard_limit,
            initial_deficit=args.initial_deficit,
            stimulus_weights=cal_stimulus_weights,
            debate_alpha=cal_debate_alpha,
            lambda_rate=cal_lambda_rate,
            run_checkpoint_path=None,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
            judge_seed=judge_seed,
        )
        metrics = _compute_metrics(logs, env.graph)
        temporal = _compute_temporal_metrics(logs, env.graph, n_windows=min(3, max(1, len([l for l in logs if l.action == "DEBATE" and l.claim]))))
        metrics.update(temporal)
        health = _detect_red_flags(mode, seed, metrics, logs)
        replay_graph = _build_aaf_from_logs(logs)
        health["replay_validation"] = {
            "live_acceptance_ratio": round(env.graph.acceptance_ratio(), 4),
            "replayed_acceptance_ratio": round(replay_graph.acceptance_ratio(), 4),
            "live_cycles": env.graph.defeat_cycle_count(),
            "replayed_cycles": replay_graph.defeat_cycle_count(),
        }
        if health["status"] == "critical":
            explanation = _explain_red_flags_with_llm(
                health,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.health_llm_model or args.model,
            )
            if explanation is not None:
                health["llm_explanation"] = explanation
        reports.append({
            "mode": mode,
            "elapsed_seconds": round(elapsed, 3),
            "metrics": metrics,
            "health": health,
        })

    report_path = preflight_dir / f"preflight_seed{seed}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    has_critical = any(r["health"]["status"] == "critical" for r in reports)
    if has_critical:
        console.print(
            f"[red]Preflight detected critical issues. Report saved to {report_path}.[/red]"
        )
        return 86

    console.print(
        f"[green]Preflight passed. Report saved to {report_path}.[/green]"
    )
    return 0


def main() -> None:
    default_api_key = os.getenv("DEEPSEEK_API_KEY", os.getenv("OPEN_AI_API_KEY", "ollama"))
    default_base_url = (
        "https://api.deepseek.com/v1" if default_api_key != "ollama"
        else "http://localhost:11434/v1"
    )
    default_model = "deepseek-v4-flash" if default_api_key != "ollama" else "llama3"

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
        choices=["epr", "epr_q", "epr_sham", "epr_no_div", "epr_llm_judge", "langgraph"],
        help="Orchestration modes to benchmark. 'epr' (Ecuación Pro-Acción Reducida) "
             "is the primary condition: endogenous homeostatic activation without Q-learning. "
             "'epr_q' is EPR + Q-learning (ablation: does the Q-learner help?). "
             "'epr_sham' controls for sigmoid shape with random δ. "
             "'epr_no_div' is EPR with the diversity term removed from g (post-hoc ablation). "
             "'epr_llm_judge' is EPR with g computed by an LLM judge online (post-hoc ablation). "
             "'langgraph' is the exogenous baseline: LLM router with access to "
             "the same structural features EPR uses internally.",
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
    parser.add_argument(
        "--no-halt-on-red-flags",
        action="store_true",
        help="Continue despite critical health alerts (not recommended for long runs).",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run a short safety check instead of the full benchmark.",
    )
    parser.add_argument(
        "--preflight-ticks",
        type=int,
        default=12,
        help="Ticks used by the short preflight run.",
    )
    parser.add_argument(
        "--health-llm-model",
        default=None,
        help="Model used to explain red flags. Defaults to the benchmark model.",
    )
    parser.add_argument(
        "--judge-models", nargs="+",
        default=["deepseek-v4-pro", "glm-5.2"],
        help="LLM models for offline judge evaluation of debate claims. "
             "Default: deepseek-v4-pro (V4-Pro) + glm-5.2.",
    )
    parser.add_argument(
        "--judge-base-urls", nargs="+",
        default=["https://api.deepseek.com/v1", "https://api.z.ai/api/paas/v4/"],
        help="OpenAI-compatible base URLs for each judge model.",
    )
    parser.add_argument(
        "--judge-api-keys", nargs="+",
        default=None,
        help="API keys for each judge model. Defaults to env vars "
             "DEEPSEEK_API_KEY and ZAI_API_KEY.",
    )
    parser.add_argument(
        "--weight-profiles", nargs="+",
        default=["uniform", "faction-heavy", "centrality-heavy",
                 "memory-heavy", "pressure-heavy"],
        help="StimulusEvaluator weight profiles for sensitivity analysis.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    health_dir = output_dir / "health_reports"
    health_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "benchmark_checkpoint.json"
    run_checkpoint_dir = output_dir / "run_checkpoints"

    if args.clean and checkpoint_path.exists():
        checkpoint_path.unlink()
        console.print("[yellow]Checkpoint cleared — starting fresh[/yellow]")
    if args.clean and run_checkpoint_dir.exists():
        for stale_checkpoint in run_checkpoint_dir.glob("*.json"):
            stale_checkpoint.unlink()

    completed = _load_bm_checkpoint(checkpoint_path)
    if completed:
        console.print(
            f"[green]Resuming: {len(completed)} seed/mode combos already completed, "
            f"skipping them.[/green]"
        )

    cal = _load_calibration_config(args.config)
    cal_stimulus_weights = cal.get("stimulus_weights")
    cal_debate_alpha = cal.get("debate_alpha", 2.0)
    cal_lambda_rate = cal.get("lambda_rate", 0.05)

    # Judge config for epr_llm_judge ablation
    judge_api_keys = args.judge_api_keys or [
        os.getenv("DEEPSEEK_API_KEY", ""),
        os.getenv("ZAI_API_KEY", ""),
    ]
    judge_model = args.judge_models[0] if args.judge_models else "deepseek-v4-pro"
    judge_base_url = args.judge_base_urls[0] if args.judge_base_urls else "https://api.deepseek.com/v1"
    judge_api_key = judge_api_keys[0] if judge_api_keys else ""
    judge_seed = 42

    if args.preflight:
        raise SystemExit(_run_preflight(args, cal))

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
            run_ck_path = _run_checkpoint_path(output_dir, mode, seed)

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
                    lambda_rate=cal_lambda_rate,
                    run_checkpoint_path=run_ck_path,
                    judge_model=judge_model,
                    judge_base_url=judge_base_url,
                    judge_api_key=judge_api_key,
                    judge_seed=judge_seed,
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
            tmp_log = log_path.with_suffix(".tmp")
            with open(tmp_log, "w", encoding="utf-8") as f:
                json.dump([e.model_dump() for e in logs], f, indent=2, ensure_ascii=False)
            tmp_log.replace(log_path)

            # Checkpoint after each (mode, seed) completes
            ck_entry = {"_ck": ck_key, "_mode": mode, "_seed": seed, **metrics}
            completed[ck_key] = ck_entry
            _save_bm_checkpoint(checkpoint_path, completed)

            health_report = _detect_red_flags(mode, seed, metrics, logs)
            health_path = health_dir / f"health_{safe_mode}_seed{seed}.json"
            if health_report["status"] != "ok":
                explanation = _explain_red_flags_with_llm(
                    health_report,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.health_llm_model or args.model,
                )
                if explanation is not None:
                    health_report["llm_explanation"] = explanation
            with open(health_path, "w", encoding="utf-8") as f:
                json.dump(health_report, f, indent=2, ensure_ascii=False)

            if health_report["status"] == "critical" and not args.no_halt_on_red_flags:
                console.print(
                    f"[red]Critical red flags detected for {mode} seed={seed}. "
                    f"Health report saved to {health_path}. Benchmark halted for review.[/red]"
                )
                raise SystemExit(86)

    # Aggregate across seeds
    agg_all = {mode: _aggregate_multi_seed(runs) for mode, runs in mode_runs.items()}

    console.rule("[bold magenta]Benchmark Results[/bold magenta]")
    _print_comparison_table(agg_all)

    # Fix 7: Run dual LLM judges on the last-seed logs for each mode
    judge_results: dict[str, Any] = {}
    judge_configs: list[dict] = []
    judge_api_keys = args.judge_api_keys or [
        os.getenv("DEEPSEEK_API_KEY", ""),
        os.getenv("ZAI_API_KEY", ""),
    ]
    for i, jm in enumerate(args.judge_models):
        judge_configs.append({
            "model": jm,
            "base_url": args.judge_base_urls[i] if i < len(args.judge_base_urls) else args.base_url,
            "api_key": judge_api_keys[i] if i < len(judge_api_keys) else args.api_key,
        })

    if last_logs and judge_configs:
        console.rule("[bold blue]LLM Judge Evaluation[/bold blue]")
        judge_results = _run_judges_for_all_modes(last_logs, judge_configs)

        for mode, jr in judge_results.items():
            avg_score = jr.get("avg_judge_score", 0.0)
            kappa = jr.get("kappa", {})
            console.print(
                f"  {mode.upper()}: avg_judge_score={avg_score:.2f}  "
                f"κ={kappa.get('kappa', 'n/a')} ({kappa.get('category', 'n/a')})"
            )

        for mode in mode_runs:
            for r in mode_runs[mode]:
                r["avg_judge_score"] = judge_results.get(mode, {}).get("avg_judge_score", 0.0)

    # Fix 4+8: Statistical tests — EPR vs LangGraph (primary comparison)
    tests = {}
    condition_mode = "epr" if "epr" in mode_runs else (
        "hrrl" if "hrrl" in mode_runs else None
    )
    baseline_mode = "langgraph" if "langgraph" in mode_runs else None

    if condition_mode and baseline_mode:
        console.rule("[bold yellow]Statistical Tests[/bold yellow]")
        tests = _run_statistical_tests(
            mode_runs[condition_mode], mode_runs[baseline_mode],
        )
        _print_statistical_tests(tests)
    elif "hrrl" in mode_runs and "langgraph" in mode_runs:
        console.rule("[bold yellow]Statistical Tests (legacy HRRL vs LG)[/bold yellow]")
        tests = _run_statistical_tests(mode_runs["hrrl"], mode_runs["langgraph"])
        _print_statistical_tests(tests)

    # Fix 9: Weight profile sensitivity (on first seed only, EPR mode)
    weight_sensitivity: dict[str, Any] = {}
    if condition_mode and last_logs.get(condition_mode):
        console.rule("[bold green]Weight Profile Sensitivity[/bold green]")
        first_seed = args.seeds[0]
        for profile_name in args.weight_profiles:
            weights = get_weight_profile(profile_name)
            console.print(f"  [dim]Profile: {profile_name} — running 1 seed...[/dim]")
            try:
                logs_p, _, env_p = _run_mode(
                    mode=condition_mode,
                    base_url=args.base_url,
                    model=args.model,
                    api_key=args.api_key,
                    iterations=args.iterations,
                    seed=first_seed,
                    api_hard_limit=args.api_hard_limit,
                    initial_deficit=args.initial_deficit,
                    stimulus_weights=weights,
                    debate_alpha=cal_debate_alpha,
                    lambda_rate=cal_lambda_rate,
                    run_checkpoint_path=None,
                )
                metrics_p = _compute_metrics(logs_p, env_p.graph)
                weight_sensitivity[profile_name] = {
                    "prr_text": metrics_p["prr_text"],
                    "prr_graph": metrics_p["prr_graph"],
                    "aaf_defeat_cycles": metrics_p["aaf_defeat_cycles"],
                    "total_debates": metrics_p["total_debates"],
                    "ir": metrics_p["ir"],
                }
                console.print(
                    f"    {profile_name}: PRR_text={metrics_p['prr_text']:.4f}  "
                    f"cycles={metrics_p['aaf_defeat_cycles']:.2f}  "
                    f"debates={metrics_p['total_debates']}"
                )
            except Exception as exc:
                weight_sensitivity[profile_name] = {"error": str(exc)}
                console.print(f"    [red]{profile_name}: failed — {exc}[/red]")

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
    if "judge_api_keys" in safe_config:
        safe_config["judge_api_keys"] = "***REDACTED***"
    report = {
        "aggregate": agg_all,
        "per_seed": mode_runs,
        "statistical_tests": tests,
        "judge_results": judge_results,
        "weight_sensitivity": weight_sensitivity,
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
