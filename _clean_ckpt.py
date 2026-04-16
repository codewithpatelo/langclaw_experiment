"""Selective cleanup of benchmark_checkpoint.json.

Removes ONLY hrrl entries (4 invalid runs). Keeps langgraph (3 valid runs).
"""
import json
from pathlib import Path

CK = Path("benchmark_results_v7/benchmark_checkpoint.json")

with open(CK, "r", encoding="utf-8") as f:
    entries = json.load(f)

before = len(entries)
print(f"Before: {before} entries")
for e in entries:
    print(f"  KEEP? {e.get('_mode')}__seed{e.get('_seed')}: {'NO (drop)' if e.get('_mode')=='hrrl' else 'YES'}")

kept = [e for e in entries if e.get("_mode") != "hrrl"]
removed = before - len(kept)

with open(CK, "w", encoding="utf-8") as f:
    json.dump(kept, f, indent=2, ensure_ascii=False)

print(f"\nAfter: {len(kept)} entries (removed {removed} hrrl entries)")
print("Remaining:")
for e in kept:
    print(f"  {e.get('_mode')}__seed{e.get('_seed')}")
