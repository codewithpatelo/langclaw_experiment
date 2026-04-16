import json
from pathlib import Path

ck = json.load(open('benchmark_results_v7/benchmark_checkpoint.json','r',encoding='utf-8'))
print(f"Checkpoint entries: {len(ck)}")
print(f"\nFirst entry keys: {list(ck[0].keys())}")
print(f"\nFirst entry sample (truncated):")
e0 = ck[0]
for k, v in e0.items():
    if isinstance(v, (dict, list)):
        print(f"  {k}: {type(v).__name__} (len={len(v)}) keys={list(v.keys())[:8] if isinstance(v, dict) else 'list'}")
    else:
        print(f"  {k}: {v}")

print("\n--- Per-run summary ---")
for e in ck:
    mode = e.get('_mode')
    seed = e.get('_seed')
    sdd = e.get('stimulus_driven_debates', e.get('SDD', '?'))
    td = e.get('total_debates', '?')
    ir = e.get('initiative_ratio', e.get('IR', '?'))
    azd = e.get('agents_with_zero_debates', '?')
    print(f"  {mode}__seed{seed}: SDD={sdd}, total_debates={td}, IR={ir}, zero={azd}")
