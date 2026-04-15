"""Hyperparameter calibration via micro-simulation ablation study.

Runs a grid of HRRL-only micro-simulations (15 ticks, 1 seed) over
two hyperparameter axes:

  1. StimulusEvaluator weight profiles  (5 configs)
  2. DEBATE satiation alpha             (3 values)

Total: 5 x 3 = 15 micro-runs.

Selection criterion: highest mean delta-phi* across debates, with
consistency rate (fraction of debates with delta_phi > 0) as tiebreaker.

Cost estimate: ~15 ticks x ~5 LLM calls/tick = ~75 calls per run.
15 runs x 75 calls x ~$0.001/call ≈ $1.10 total.

Output: calibration_results.json with ranked configs and the best one.

Usage:
    python calibrate_hyperparams.py
    python calibrate_hyperparams.py --ticks 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from langclaw.simulation import OrchestrationMode, SotopiaEnvironment

load_dotenv()
console = Console()

WEIGHT_CONFIGS: dict[str, dict[str, float]] = {
    "equal": {
        "w_faction": 0.20, "w_centrality": 0.20,
        "w_memory": 0.20, "w_novelty": 0.20, "w_pressure": 0.20,
    },
    "faction_dominant": {
        "w_faction": 0.40, "w_centrality": 0.15,
        "w_memory": 0.15, "w_novelty": 0.15, "w_pressure": 0.15,
    },
    "centrality_dominant": {
        "w_faction": 0.15, "w_centrality": 0.40,
        "w_memory": 0.15, "w_novelty": 0.15, "w_pressure": 0.15,
    },
    "novelty_dominant": {
        "w_faction": 0.15, "w_centrality": 0.15,
        "w_memory": 0.15, "w_novelty": 0.40, "w_pressure": 0.15,
    },
    "pressure_dominant": {
        "w_faction": 0.15, "w_centrality": 0.15,
        "w_memory": 0.15, "w_novelty": 0.15, "w_pressure": 0.40,
    },
}

ALPHA_VALUES = [1.0, 2.0, 3.0]


def _run_micro(
    base_url: str,
    model: str,
    api_key: str,
    ticks: int,
    seed: int,
    stimulus_weights: dict[str, float],
    debate_alpha: float,
    api_hard_limit: int,
) -> dict:
    """Run a single micro-simulation and return summary metrics."""
    env = SotopiaEnvironment(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_iterations=ticks,
        seed=seed,
        orchestration_mode=OrchestrationMode.HRRL,
        api_hard_limit=api_hard_limit,
        stimulus_weights=stimulus_weights,
        debate_alpha=debate_alpha,
    )

    t0 = time.perf_counter()
    logs = env.run()
    elapsed = time.perf_counter() - t0

    debates = [l for l in logs if l.action == "DEBATE"]
    n_debates = len(debates)
    dphi_values = [l.delta_phi for l in debates]

    avg_dphi = sum(dphi_values) / n_debates if n_debates > 0 else 0.0
    consistency = (
        sum(1 for d in dphi_values if d > 0) / n_debates
        if n_debates > 0 else 0.0
    )

    return {
        "n_debates": n_debates,
        "avg_dphi": round(avg_dphi, 6),
        "consistency": round(consistency, 4),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    default_api_key = os.getenv("OPEN_AI_API_KEY", "ollama")
    default_base_url = (
        "https://api.openai.com/v1" if default_api_key != "ollama"
        else "http://localhost:11434/v1"
    )
    default_model = "gpt-5-nano" if default_api_key != "ollama" else "llama3"

    parser = argparse.ArgumentParser(
        description="LangClaw Hyperparameter Calibration (ablation micro-simulation)"
    )
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--api-key", default=default_api_key)
    parser.add_argument("--ticks", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-hard-limit", type=int, default=200)
    parser.add_argument("--output", default="calibration_results.json")
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    results: list[dict] = []
    total = len(WEIGHT_CONFIGS) * len(ALPHA_VALUES)
    run_idx = 0

    for w_name, w_cfg in WEIGHT_CONFIGS.items():
        for alpha in ALPHA_VALUES:
            run_idx += 1
            console.rule(
                f"[bold cyan]Run {run_idx}/{total}: "
                f"weights={w_name}, alpha={alpha}[/bold cyan]"
            )

            metrics = _run_micro(
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                ticks=args.ticks,
                seed=args.seed,
                stimulus_weights=w_cfg,
                debate_alpha=alpha,
                api_hard_limit=args.api_hard_limit,
            )

            console.print(
                f"  debates={metrics['n_debates']}, "
                f"avg_dphi={metrics['avg_dphi']:.4f}, "
                f"consistency={metrics['consistency']:.2%}, "
                f"time={metrics['elapsed_s']:.1f}s"
            )

            results.append({
                "weight_config_name": w_name,
                "stimulus_weights": w_cfg,
                "debate_alpha": alpha,
                **metrics,
            })

    results.sort(key=lambda r: (r["avg_dphi"], r["consistency"]), reverse=True)
    best = results[0]

    table = Table(
        title="Calibration Results (ranked by avg delta-phi)",
        show_lines=True,
        title_style="bold magenta",
    )
    table.add_column("Rank", justify="right", width=5)
    table.add_column("Weights", width=22)
    table.add_column("Alpha", justify="right", width=6)
    table.add_column("Debates", justify="right", width=8)
    table.add_column("Avg dphi", justify="right", width=10)
    table.add_column("Consistency", justify="right", width=12)

    for i, r in enumerate(results, 1):
        style = "bold green" if i == 1 else ""
        table.add_row(
            str(i),
            r["weight_config_name"],
            str(r["debate_alpha"]),
            str(r["n_debates"]),
            f"{r['avg_dphi']:.4f}",
            f"{r['consistency']:.2%}",
            style=style,
        )

    console.print(table)
    console.print(
        f"\n[bold green]Best config:[/bold green] weights={best['weight_config_name']}, "
        f"alpha={best['debate_alpha']}, avg_dphi={best['avg_dphi']:.4f}"
    )

    output = {
        "best_config": {
            "stimulus_weights": best["stimulus_weights"],
            "debate_alpha": best["debate_alpha"],
            "weight_config_name": best["weight_config_name"],
        },
        "selection_criteria": "max avg_dphi, tiebreak by consistency",
        "micro_sim_params": {
            "ticks": args.ticks,
            "seed": args.seed,
            "mode": "hrrl",
            "api_hard_limit": args.api_hard_limit,
        },
        "all_configs": results,
    }

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Results saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
