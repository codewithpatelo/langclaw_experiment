"""CLI entry point for running the LangClaw simulation.

Usage examples
--------------
# HRRL mode (default — endogenous homeostatic regulation):
    python main.py --orchestration-mode hrrl --output results_hrrl.json

# Baseline: round-robin (every agent speaks every tick):
    python main.py --orchestration-mode round-robin --output results_rr.json

# Baseline: random agent each tick:
    python main.py --orchestration-mode random --output results_random.json

# With Ollama (local):
    python main.py --base-url http://localhost:11434/v1 --model llama3 --api-key ollama

# Override model and iterations:
    python main.py --model gpt-4o --iterations 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

from langclaw.simulation import OrchestrationMode, SotopiaEnvironment

load_dotenv()


def main() -> None:
    default_api_key = os.getenv("OPEN_AI_API_KEY", "ollama")
    default_base_url = (
        "https://api.openai.com/v1" if default_api_key != "ollama"
        else "http://localhost:11434/v1"
    )
    default_model = "gpt-4o-mini" if default_api_key != "ollama" else "llama3"

    parser = argparse.ArgumentParser(
        description="LangClaw — Homeostatic Multi-Agent Debate Simulation",
    )
    parser.add_argument(
        "--base-url",
        default=default_base_url,
        help=f"OpenAI-compatible API base URL (default: {default_base_url})",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"Model name (default: {default_model})",
    )
    parser.add_argument(
        "--api-key",
        default=default_api_key,
        help="API key (default: from .env OPEN_AI_API_KEY)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of simulation ticks (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--orchestration-mode",
        default="hrrl",
        choices=["hrrl", "round-robin", "random"],
        help="Orchestration mode: hrrl (default) | round-robin | random",
    )
    parser.add_argument(
        "--api-hard-limit",
        type=int,
        default=200,
        help="Maximum total API calls per agent per run (default: 200)",
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=0.0,
        help="Seconds of wall-clock time between ticks (0 = instant, default: 0.0). "
             "Makes ticks real-time pulses instead of tight iterations.",
    )
    parser.add_argument(
        "--initial-deficit",
        type=float,
        default=0.5,
        help="Starting epistemic deficit for all agents (default: 0.5). "
             "At theta=0.7 and lambda=0.05/tick, 50%% activation at tick ~4.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the simulation log as JSON",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    env = SotopiaEnvironment(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_iterations=args.iterations,
        seed=args.seed,
        orchestration_mode=OrchestrationMode(args.orchestration_mode),
        api_hard_limit=args.api_hard_limit,
        tick_interval=args.tick_interval,
        initial_deficit=args.initial_deficit,
    )

    logs = env.run()

    if args.output:
        data = [entry.model_dump() for entry in logs]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nLog saved to {args.output}")


if __name__ == "__main__":
    main()
