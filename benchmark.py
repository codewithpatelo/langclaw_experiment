"""Comparative benchmark: HRRL vs Round-Robin orchestration.

Runs the simulation under each orchestration mode with the same master seed,
collects per-tick logs, and generates a summary report with comparison tables
and a set of Plotly charts saved as HTML.

The benchmark supports debate-budget matching: HRRL runs for T ticks and
produces N debates organically.  Round-Robin then runs for ceil(N / n_agents)
ticks so that both modes consume approximately the same debate budget.
Multiple seeds can be specified to obtain mean +/- std across runs.

Usage
-----
    python benchmark.py --model gpt-4o-mini --iterations 50 --seeds 42 123 256
    python benchmark.py --modes hrrl round-robin --output-dir results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from langclaw.schemas import SimulationLog
from langclaw.simulation import OrchestrationMode, SotopiaEnvironment

load_dotenv()
console = Console()

MODES = ["hrrl", "round-robin"]


def _run_mode(
    mode: str,
    base_url: str,
    model: str,
    api_key: str,
    iterations: int,
    seed: int | None,
    api_hard_limit: int,
    initial_deficit: float,
) -> tuple[list[SimulationLog], float]:
    """Run a full simulation for the given mode and return (logs, elapsed_seconds)."""
    env = SotopiaEnvironment(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_iterations=iterations,
        seed=seed,
        orchestration_mode=OrchestrationMode(mode),
        api_hard_limit=api_hard_limit,
        initial_deficit=initial_deficit,
    )
    t0 = time.perf_counter()
    logs = env.run()
    elapsed = time.perf_counter() - t0
    return logs, elapsed


def _compute_metrics(logs: list[SimulationLog]) -> dict:
    """Derive aggregate metrics from a set of simulation logs."""
    debates = [l for l in logs if l.action == "DEBATE"]
    total_ticks = max((l.tick for l in logs), default=0)
    agents = sorted(set(l.agent_id for l in logs))

    consistent = sum(1 for l in debates if l.target_node_id is not None)
    tau = consistent / len(debates) if debates else 0.0

    final_deficits = {}
    for agent_id in agents:
        agent_logs = [l for l in logs if l.agent_id == agent_id]
        if agent_logs:
            final_deficits[agent_id] = agent_logs[-1].deficit_after

    avg_delta_phi = sum(l.delta_phi for l in debates) / len(debates) if debates else 0.0

    graph_nodes = logs[-1].graph_nodes if logs else 0
    graph_edges = logs[-1].graph_edges if logs else 0

    per_agent_debates = {}
    per_agent_avg_dphi = {}
    for agent_id in agents:
        ad = [l for l in debates if l.agent_id == agent_id]
        per_agent_debates[agent_id] = len(ad)
        per_agent_avg_dphi[agent_id] = (
            sum(l.delta_phi for l in ad) / len(ad) if ad else 0.0
        )

    return {
        "total_ticks": total_ticks,
        "total_debates": len(debates),
        "tau_bench": round(tau, 4),
        "avg_delta_phi": round(avg_delta_phi, 4),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "final_deficits": {k: round(v, 4) for k, v in final_deficits.items()},
        "per_agent_debates": per_agent_debates,
        "per_agent_avg_dphi": {k: round(v, 4) for k, v in per_agent_avg_dphi.items()},
    }


def _save_comparison_charts(all_metrics: dict, all_logs: dict, output_dir: Path) -> None:
    """Generate Plotly charts comparing modes and save as HTML."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        console.print("[yellow]plotly not installed — skipping chart generation[/yellow]")
        return

    modes = list(all_metrics.keys())

    # 1. Bar chart: τ-bench comparison
    fig_tau = go.Figure(data=[
        go.Bar(
            x=modes,
            y=[all_metrics[m]["tau_bench"] for m in modes],
            marker_color=["#2ecc71", "#3498db"][:len(modes)],
            text=[f'{all_metrics[m]["tau_bench"]:.3f}' for m in modes],
            textposition="auto",
        )
    ])
    fig_tau.update_layout(
        title="τ-bench (Logical Consistency) by Orchestration Mode",
        yaxis_title="τ-bench",
        template="plotly_dark",
    )
    fig_tau.write_html(str(output_dir / "tau_bench_comparison.html"))

    # 2. Bar chart: total debates and avg Δφ*
    fig_debates = make_subplots(rows=1, cols=2, subplot_titles=["Total Debates", "Avg Δφ*"])
    colors = ["#2ecc71", "#3498db"][:len(modes)]
    fig_debates.add_trace(
        go.Bar(x=modes, y=[all_metrics[m]["total_debates"] for m in modes],
               marker_color=colors, showlegend=False),
        row=1, col=1,
    )
    fig_debates.add_trace(
        go.Bar(x=modes, y=[all_metrics[m]["avg_delta_phi"] for m in modes],
               marker_color=colors, showlegend=False),
        row=1, col=2,
    )
    fig_debates.update_layout(title="Debate Volume and Quality", template="plotly_dark")
    fig_debates.write_html(str(output_dir / "debates_comparison.html"))

    # 3. Line chart: deficit evolution per agent per mode
    fig_deficit = make_subplots(
        rows=1, cols=len(modes),
        subplot_titles=[f"Mode: {m}" for m in modes],
        shared_yaxes=True,
    )
    agent_colors = {"GOV-1": "#2ecc71", "GOV-2": "#27ae60", "OPP-1": "#e74c3c", "OPP-2": "#c0392b"}
    for col_idx, mode in enumerate(modes, start=1):
        logs = all_logs[mode]
        agents = sorted(set(l.agent_id for l in logs))
        for agent_id in agents:
            agent_logs = [l for l in logs if l.agent_id == agent_id]
            fig_deficit.add_trace(
                go.Scatter(
                    x=[l.tick for l in agent_logs],
                    y=[l.deficit_after for l in agent_logs],
                    mode="lines",
                    name=f"{agent_id}" if col_idx == 1 else None,
                    line=dict(color=agent_colors.get(agent_id, "#95a5a6")),
                    showlegend=(col_idx == 1),
                ),
                row=1, col=col_idx,
            )
    fig_deficit.update_layout(
        title="Epistemic Deficit Evolution by Mode",
        template="plotly_dark",
        height=400,
    )
    fig_deficit.write_html(str(output_dir / "deficit_evolution.html"))

    # 4. Per-agent debate contribution stacked bar
    fig_agent = go.Figure()
    all_agents = sorted(set(
        a for m in modes for a in all_metrics[m]["per_agent_debates"]
    ))
    for agent_id in all_agents:
        fig_agent.add_trace(go.Bar(
            x=modes,
            y=[all_metrics[m]["per_agent_debates"].get(agent_id, 0) for m in modes],
            name=agent_id,
            marker_color=agent_colors.get(agent_id, "#95a5a6"),
        ))
    fig_agent.update_layout(
        barmode="stack",
        title="Debates per Agent by Mode",
        yaxis_title="Number of DEBATE actions",
        template="plotly_dark",
    )
    fig_agent.write_html(str(output_dir / "agent_debates.html"))

    console.print(f"  [green]Charts saved to {output_dir}/[/green]")


def _print_comparison_table(all_metrics: dict) -> None:
    """Print a Rich comparison table to the console."""
    table = Table(title="Benchmark Comparison", show_lines=True, title_style="bold magenta")
    table.add_column("Metric", style="cyan", width=24)
    for mode in all_metrics:
        table.add_column(mode.upper(), justify="right", width=14)

    rows = [
        ("Total Debates", lambda m: str(m["total_debates"])),
        ("τ-bench", lambda m: f'{m["tau_bench"]:.4f}'),
        ("Avg Δφ*", lambda m: f'{m["avg_delta_phi"]:.4f}'),
        ("Graph Nodes", lambda m: str(m["graph_nodes"])),
        ("Graph Edges", lambda m: str(m["graph_edges"])),
    ]
    for label, fn in rows:
        table.add_row(label, *(fn(all_metrics[mode]) for mode in all_metrics))

    # Per-agent rows
    agents = sorted(set(
        a for m in all_metrics.values() for a in m["per_agent_debates"]
    ))
    for agent_id in agents:
        table.add_row(
            f"  {agent_id} debates",
            *(str(all_metrics[mode]["per_agent_debates"].get(agent_id, 0)) for mode in all_metrics),
        )
        table.add_row(
            f"  {agent_id} avg Δφ*",
            *(f'{all_metrics[mode]["per_agent_avg_dphi"].get(agent_id, 0.0):.4f}' for mode in all_metrics),
        )
    for agent_id in agents:
        table.add_row(
            f"  {agent_id} final δ",
            *(f'{all_metrics[mode]["final_deficits"].get(agent_id, 0.0):.4f}' for mode in all_metrics),
        )

    console.print(table)


import math
import statistics


def _aggregate_multi_seed(runs: list[dict]) -> dict:
    """Compute mean +/- std across multiple seeded runs."""
    numeric_keys = ["total_debates", "tau_bench", "avg_delta_phi", "graph_nodes", "graph_edges"]
    agg: dict[str, Any] = {}
    for key in numeric_keys:
        vals = [r[key] for r in runs]
        agg[key] = round(statistics.mean(vals), 4)
        agg[f"{key}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0

    agents = sorted(set(a for r in runs for a in r["per_agent_debates"]))
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


def main() -> None:
    default_api_key = os.getenv("OPEN_AI_API_KEY", "ollama")
    default_base_url = (
        "https://api.openai.com/v1" if default_api_key != "ollama"
        else "http://localhost:11434/v1"
    )
    default_model = "gpt-4o-mini" if default_api_key != "ollama" else "llama3"

    parser = argparse.ArgumentParser(description="LangClaw Benchmark -- HRRL vs Round-Robin")
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--api-key", default=default_api_key)
    parser.add_argument("--iterations", type=int, default=50,
                        help="Max ticks for HRRL. Round-Robin ticks auto-computed to match debate budget.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 256],
                        help="Seeds for multi-run statistical analysis")
    parser.add_argument("--api-hard-limit", type=int, default=200)
    parser.add_argument("--initial-deficit", type=float, default=0.5)
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_agents = 4
    hrrl_runs: list[dict] = []
    rr_runs: list[dict] = []
    last_hrrl_logs: list[SimulationLog] = []
    last_rr_logs: list[SimulationLog] = []

    for seed in args.seeds:
        console.rule(f"[bold yellow]Seed: {seed}[/bold yellow]")

        # 1) Run HRRL for full T ticks
        console.rule(f"[bold cyan]Running HRRL (T={args.iterations}, seed={seed})[/bold cyan]")
        hrrl_logs, hrrl_time = _run_mode(
            mode="hrrl",
            base_url=args.base_url, model=args.model, api_key=args.api_key,
            iterations=args.iterations, seed=seed,
            api_hard_limit=args.api_hard_limit, initial_deficit=args.initial_deficit,
        )
        hrrl_metrics = _compute_metrics(hrrl_logs)
        hrrl_debates = hrrl_metrics["total_debates"]
        console.print(f"  HRRL produced {hrrl_debates} debates in {hrrl_time:.1f}s")

        # 2) Run Round-Robin with matched debate budget
        rr_ticks = max(1, math.ceil(hrrl_debates / n_agents))
        console.rule(
            f"[bold cyan]Running Round-Robin (T={rr_ticks}, targeting ~{hrrl_debates} debates, seed={seed})[/bold cyan]"
        )
        rr_logs, rr_time = _run_mode(
            mode="round-robin",
            base_url=args.base_url, model=args.model, api_key=args.api_key,
            iterations=rr_ticks, seed=seed,
            api_hard_limit=args.api_hard_limit, initial_deficit=args.initial_deficit,
        )
        rr_metrics = _compute_metrics(rr_logs)
        console.print(f"  Round-Robin produced {rr_metrics['total_debates']} debates in {rr_time:.1f}s\n")

        hrrl_runs.append(hrrl_metrics)
        rr_runs.append(rr_metrics)
        last_hrrl_logs = hrrl_logs
        last_rr_logs = rr_logs

        # Save per-seed logs
        for mode_name, logs in [("hrrl", hrrl_logs), ("round_robin", rr_logs)]:
            log_path = output_dir / f"logs_{mode_name}_seed{seed}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([e.model_dump() for e in logs], f, indent=2, ensure_ascii=False)

    # Aggregate across seeds
    agg_hrrl = _aggregate_multi_seed(hrrl_runs)
    agg_rr = _aggregate_multi_seed(rr_runs)
    all_metrics = {"hrrl": agg_hrrl, "round-robin": agg_rr}

    console.rule("[bold magenta]Benchmark Results (mean across seeds)[/bold magenta]")
    _print_comparison_table(all_metrics)

    # Print std info
    console.print(f"\n[bold]Seeds used:[/bold] {args.seeds}")
    console.print(f"[bold]HRRL debates per seed:[/bold] {[r['total_debates'] for r in hrrl_runs]}")
    console.print(f"[bold]RR debates per seed:[/bold]   {[r['total_debates'] for r in rr_runs]}")
    console.print(f"\n[bold]Std dev:[/bold]")
    for key in ["tau_bench", "avg_delta_phi"]:
        console.print(f"  HRRL {key}: {agg_hrrl.get(f'{key}_std', 0):.4f}")
        console.print(f"  RR   {key}: {agg_rr.get(f'{key}_std', 0):.4f}")

    # Save report
    report = {
        "aggregate": all_metrics,
        "per_seed": {"hrrl": hrrl_runs, "round-robin": rr_runs},
        "config": vars(args),
    }
    with open(output_dir / "benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    all_logs = {"hrrl": last_hrrl_logs, "round-robin": last_rr_logs}
    _save_comparison_charts(all_metrics, all_logs, output_dir)

    console.print(f"\n[green]All results saved to {output_dir}/[/green]")


if __name__ == "__main__":
    main()
