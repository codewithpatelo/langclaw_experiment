"""Parallel experiment launcher — 4 lots with separate API keys.

Each lot runs 5 seeds × 4 modes × 80 ticks independently.
Results accumulate in separate output dirs, merged afterward.

Usage:
    python run_parallel.py
    python run_parallel.py --iterations 80
    python run_parallel.py --dry-run  # print commands without executing

Resume: just re-run the same command. Each lot resumes from its own checkpoint.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from langclaw.seeds import SeedFactory

# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

EXPERIMENT_MASTER_SEED = 20260308
SEEDS = SeedFactory.derive_experiment_seeds(EXPERIMENT_MASTER_SEED, n=20)

LOTS = [
    {"name": "lot1", "seeds": SEEDS[0:5],   "key_env": "DEEPSEEK_API_KEY_1"},
    {"name": "lot2", "seeds": SEEDS[5:10],  "key_env": "DEEPSEEK_API_KEY_2"},
    {"name": "lot3", "seeds": SEEDS[10:15], "key_env": "DEEPSEEK_API_KEY_3"},
    {"name": "lot4", "seeds": SEEDS[15:20], "key_env": "DEEPSEEK_API_KEY_4"},
]

BASE_DIR = Path(__file__).parent
OUTPUT_BASE = BASE_DIR / "experiment_results"


def load_env() -> dict[str, str]:
    """Load .env file manually (since python-dotenv may not be installed)."""
    env_path = BASE_DIR / ".env"
    env_vars: dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                env_vars[key] = val
    return env_vars


def build_command(lot: dict, api_key: str, iterations: int, judge_keys: list[str]) -> list[str]:
    """Build the benchmark command for a single lot."""
    output_dir = OUTPUT_BASE / lot["name"]
    seeds_str = " ".join(str(s) for s in lot["seeds"])

    cmd = [
        sys.executable, str(BASE_DIR / "benchmark.py"),
        "--iterations", str(iterations),
        "--seeds", *[str(s) for s in lot["seeds"]],
        "--modes", "epr", "epr_q", "epr_sham", "langgraph",
        "--output-dir", str(output_dir),
        "--api-key", api_key,
        "--model", "deepseek-v4-flash",
        "--config", str(BASE_DIR / "recalibration_results.json"),
        "--judge-models", "deepseek-v4-pro", "glm-5.2",
        "--judge-base-urls", "https://api.deepseek.com/v1", "https://api.z.ai/api/paas/v4/",
        "--judge-api-keys", judge_keys[0] if judge_keys else api_key, judge_keys[1] if len(judge_keys) > 1 else "",
        "--log-level", "INFO",
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel experiment launcher")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--lot", type=int, choices=[1, 2, 3, 4],
                        help="Run only one lot (for testing)")
    args = parser.parse_args()

    env_vars = load_env()

    # Validate API keys
    missing_keys = []
    for lot in LOTS:
        key = env_vars.get(lot["key_env"])
        if not key:
            missing_keys.append(lot["key_env"])
    if missing_keys:
        print(f"ERROR: Missing API keys in .env: {', '.join(missing_keys)}")
        sys.exit(1)

    # Judge keys — DeepSeek + GLM
    judge_keys = [
        env_vars.get("DEEPSEEK_API_KEY_1", ""),
        env_vars.get("ZAI_API_KEY", ""),
    ]

    # Select lots to run
    lots_to_run = LOTS if args.lot is None else [LOTS[args.lot - 1]]

    # Print commands
    print("=" * 70)
    print("PARALLEL EXPERIMENT LAUNCHER")
    print("=" * 70)
    for lot in lots_to_run:
        key = env_vars[lot["key_env"]]
        cmd = build_command(lot, key, args.iterations, judge_keys)
        print(f"\n[{lot['name']}] seeds={lot['seeds']}")
        print(f"  key={lot['key_env']} ({key[:8]}...)")
        print(f"  output={OUTPUT_BASE / lot['name']}")
        print(f"  cmd={' '.join(cmd)}")

    if args.dry_run:
        print("\n[DRY RUN] No commands executed.")
        return

    # Launch processes
    print("\n" + "=" * 70)
    print("LAUNCHING PARALLEL PROCESSES")
    print("=" * 70)

    processes: list[subprocess.Popen] = []
    for lot in lots_to_run:
        key = env_vars[lot["key_env"]]
        cmd = build_command(lot, key, args.iterations, judge_keys)

        # Pass env vars to subprocess
        lot_env = os.environ.copy()
        lot_env["DEEPSEEK_API_KEY"] = key
        lot_env["OPEN_AI_API_KEY"] = env_vars.get("OPEN_AI_API_KEY", lot_env.get("OPEN_AI_API_KEY", ""))
        lot_env["ZAI_API_KEY"] = env_vars.get("ZAI_API_KEY", "")

        log_path = OUTPUT_BASE / lot["name"] / "process.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            env=lot_env,
        )
        processes.append(proc)
        print(f"  [{lot['name']}] PID={proc.pid} -> {log_path}")

    print(f"\n{len(processes)} processes launched. Monitoring...")

    # Monitor processes
    start_time = time.time()
    while True:
        time.sleep(30)
        elapsed = time.time() - start_time
        all_done = True
        for i, proc in enumerate(processes):
            lot_name = lots_to_run[i]["name"]
            if proc.poll() is None:
                all_done = False
                print(f"  [{elapsed/60:.0f}m] {lot_name}: RUNNING (PID={proc.pid})")
            else:
                rc = proc.returncode
                if rc == 0:
                    print(f"  [{elapsed/60:.0f}m] {lot_name}: DONE (exit=0)")
                elif rc == 75:
                    print(f"  [{elapsed/60:.0f}m] {lot_name}: PAUSED (rate limit, exit=75) — re-run to resume")
                else:
                    print(f"  [{elapsed/60:.0f}m] {lot_name}: FAILED (exit={rc})")

        if all_done:
            break

    print("\n" + "=" * 70)
    print("ALL PROCESSES FINISHED")
    print("=" * 70)
    for i, proc in enumerate(processes):
        lot_name = lots_to_run[i]["name"]
        rc = proc.returncode
        status = "DONE" if rc == 0 else ("PAUSED (rate limit)" if rc == 75 else f"FAILED (exit={rc})")
        print(f"  {lot_name}: {status}")

    print(f"\nResults in: {OUTPUT_BASE}/")
    print("Merge with: python merge_results.py")


if __name__ == "__main__":
    main()
