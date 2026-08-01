"""Merge results from parallel experiment lots.

Reads benchmark_checkpoint.json + per-seed log files from each lot directory,
combines them into a single dataset, and writes merged output.

Usage:
    python merge_results.py
    python merge_results.py --output-dir experiment_results --output-file merged_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).parent
DEFAULT_INPUT = BASE_DIR / "experiment_results"
DEFAULT_OUTPUT = DEFAULT_INPUT / "merged"


def merge_checkpoints(input_dir: Path) -> dict:
    """Merge benchmark_checkpoint.json from all lot directories."""
    merged: dict[str, dict] = {}

    for lot_dir in sorted(input_dir.glob("lot*")):
        ck_path = lot_dir / "benchmark_checkpoint.json"
        if not ck_path.exists():
            print(f"  [skip] {lot_dir.name}: no checkpoint file")
            continue

        with open(ck_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        if isinstance(entries, list):
            for entry in entries:
                key = entry.get("_ck", f"{entry.get('_mode', '?')}__seed{entry.get('_seed', '?')}")
                if key in merged:
                    print(f"  [warn] duplicate key {key} in {lot_dir.name}")
                merged[key] = entry
            print(f"  [ok] {lot_dir.name}: {len(entries)} entries")

    return merged


def merge_logs(input_dir: Path, output_dir: Path) -> None:
    """Copy per-seed log files from all lots into merged directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for lot_dir in sorted(input_dir.glob("lot*")):
        logs_dir = lot_dir / "logs"
        if not logs_dir.exists():
            continue

        for log_file in logs_dir.glob("*.json"):
            dest = output_dir / log_file.name
            if dest.exists():
                print(f"  [warn] {log_file.name} already exists, overwriting")
            dest.write_text(log_file.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  [ok] {log_file.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge parallel experiment results")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print("=" * 60)
    print("MERGING EXPERIMENT RESULTS")
    print("=" * 60)

    # Merge checkpoints
    print("\n[1/2] Merging checkpoints...")
    merged = merge_checkpoints(args.input_dir)

    # Write merged checkpoint
    merged_ck_path = args.output_dir / "benchmark_checkpoint.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(merged_ck_path, "w", encoding="utf-8") as f:
        json.dump(list(merged.values()), f, indent=2, ensure_ascii=False)
    print(f"\n  Merged checkpoint: {merged_ck_path} ({len(merged)} entries)")

    # Merge logs
    print("\n[2/2] Merging logs...")
    merge_logs(args.input_dir, args.output_dir / "logs")

    # Summary
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)

    # Per-mode summary
    modes: dict[str, list[dict]] = {}
    for entry in merged.values():
        mode = entry.get("_mode", "?")
        modes.setdefault(mode, []).append(entry)

    for mode in sorted(modes):
        entries = modes[mode]
        debates = [e.get("total_debates", 0) for e in entries]
        aaf = [e.get("aaf_acceptance_ratio", 0) for e in entries]
        prr = [e.get("prr_text", 0) for e in entries]
        n = len(entries)
        avg_deb = sum(debates) / n if n else 0
        avg_aaf = sum(aaf) / n if n else 0
        avg_prr = sum(prr) / n if n else 0
        print(f"  {mode}: n={n} debates={avg_deb:.1f} aaf={avg_aaf:.3f} prr={avg_prr:.3f}")

    print(f"\n  Total entries: {len(merged)}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
