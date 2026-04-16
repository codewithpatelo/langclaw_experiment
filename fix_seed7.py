"""Remove contaminated hrrl__seed7 from benchmark checkpoint and re-run it.

Run this AFTER the current benchmark finishes all other seeds.
It removes the hrrl__seed7 entry from the checkpoint so benchmark.py treats
it as pending, then launches benchmark with --seeds 7 --modes hrrl.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

OUTPUT_DIR = Path("benchmark_results_v7")
CHECKPOINT = OUTPUT_DIR / "benchmark_checkpoint.json"
CONTAMINATED_LOG = OUTPUT_DIR / "logs_hrrl_seed7.json"
CONTAMINATED_LOG_BACKUP = OUTPUT_DIR / "logs_hrrl_seed7_CONTAMINATED.json"
RUN_CK = OUTPUT_DIR / "run_checkpoints" / "hrrl__seed7.json"


def main() -> int:
    if not CHECKPOINT.exists():
        print("No checkpoint found.")
        return 1

    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        entries = json.load(f)

    before = len(entries)
    entries = [e for e in entries if e.get("_ck") != "hrrl__seed7"]
    after = len(entries)

    if before == after:
        print("hrrl__seed7 not found in checkpoint; nothing to remove.")
        return 0

    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    if CONTAMINATED_LOG.exists():
        shutil.move(str(CONTAMINATED_LOG), str(CONTAMINATED_LOG_BACKUP))
        print(f"Backed up contaminated log to {CONTAMINATED_LOG_BACKUP}")

    if RUN_CK.exists():
        RUN_CK.unlink()
        print(f"Deleted stale run checkpoint {RUN_CK}")

    print(f"Removed hrrl__seed7 from checkpoint ({before} -> {after} entries).")
    print()
    print("Now run:")
    print("  python benchmark.py --iterations 80 --seeds 7 --modes hrrl "
          "--config calibration_results.json --output-dir benchmark_results_v7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
