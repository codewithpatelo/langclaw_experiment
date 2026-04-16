"""Complete remaining benchmark runs in two phases.

Phase 1: Finish LANGGRAPH seeds (42 resumes from tick 58, then 123, 256).
Phase 2: Remove all HRRL entries from checkpoint, then re-run HRRL for all 5 seeds
          with the fixed _hrrl_single_tick (broadcast + messaging).

Usage:
    python run_remaining.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("benchmark_results_v7")
CHECKPOINT = OUTPUT_DIR / "benchmark_checkpoint.json"
RUN_CK_DIR = OUTPUT_DIR / "run_checkpoints"
SEEDS = [7, 17, 42, 123, 256]
ITERATIONS = 80
CONFIG = "calibration_results.json"


def _run_benchmark(modes: str, seeds: list[int], label: str) -> int:
    seed_str = " ".join(str(s) for s in seeds)
    cmd = [
        sys.executable, "benchmark.py",
        "--iterations", str(ITERATIONS),
        "--seeds", *[str(s) for s in seeds],
        "--modes", modes,
        "--config", CONFIG,
        "--output-dir", str(OUTPUT_DIR),
    ]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    return result.returncode


def _clean_hrrl_from_checkpoint() -> None:
    if not CHECKPOINT.exists():
        print("No checkpoint found, nothing to clean.")
        return

    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        entries = json.load(f)

    before = len(entries)
    kept = [e for e in entries if e.get("_mode") != "hrrl"]
    removed = before - len(kept)

    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print(f"Removed {removed} HRRL entries from checkpoint ({before} -> {len(kept)}).")

    for seed in SEEDS:
        ck = RUN_CK_DIR / f"hrrl__seed{seed}.json"
        if ck.exists():
            ck.unlink()
            print(f"  Deleted stale run checkpoint: {ck.name}")

    for seed in SEEDS:
        log = OUTPUT_DIR / f"logs_hrrl_seed{seed}.json"
        if log.exists():
            backup = OUTPUT_DIR / f"logs_hrrl_seed{seed}_OLD_NOBROADCAST.json"
            shutil.move(str(log), str(backup))
            print(f"  Backed up old log: {log.name} -> {backup.name}")


def main() -> int:
    print("Phase 1: Complete remaining LANGGRAPH runs")
    print("  seed42 will resume from tick 58 (run checkpoint exists)")
    print("  seeds 123, 256 will run from scratch")
    print("-" * 60)

    rc = _run_benchmark("langgraph", [42, 123, 256], "LANGGRAPH seeds 42, 123, 256")
    if rc != 0:
        print(f"\nPhase 1 FAILED with exit code {rc}. Aborting.")
        return rc

    lg_seeds_done = set()
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            if e.get("_mode") == "langgraph":
                lg_seeds_done.add(e["_seed"])

    missing_lg = set(SEEDS) - lg_seeds_done
    if missing_lg:
        print(f"\nWARNING: LANGGRAPH seeds {missing_lg} still missing. Running them.")
        rc = _run_benchmark("langgraph", sorted(missing_lg), f"LANGGRAPH missing seeds {missing_lg}")
        if rc != 0:
            print(f"\nPhase 1 (retry) FAILED with exit code {rc}. Aborting.")
            return rc

    print("\n" + "=" * 60)
    print("Phase 1 COMPLETE - All LANGGRAPH runs done.")
    print("=" * 60)

    print("\nPhase 2: Clean HRRL from checkpoint and re-run all 5 seeds")
    print("-" * 60)
    _clean_hrrl_from_checkpoint()

    rc = _run_benchmark("hrrl", SEEDS, "HRRL all seeds (with broadcast fix)")
    if rc != 0:
        print(f"\nPhase 2 FAILED with exit code {rc}.")
        return rc

    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
