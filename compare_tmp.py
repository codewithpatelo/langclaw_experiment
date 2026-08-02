import json, glob, os

def load_logs(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def extract_metrics(logs):
    debates = [l for l in logs if l.get("action") == "DEBATE" and l.get("claim")]
    g_values = [l.get("delta_phi", 0) for l in debates]
    max_g = max(g_values) if g_values else 0
    avg_g = sum(g_values) / len(g_values) if g_values else 0
    agent_counts = {}
    for l in debates:
        aid = l.get("agent_id", "?")
        agent_counts[aid] = agent_counts.get(aid, 0) + 1
    total = len(debates)
    max_agent_share = max(agent_counts.values()) / total if total else 0
    judge_debates = [l for l in debates if l.get("judge_info")]
    neg_g = sum(1 for g in g_values if g < 0)
    return {
        "n_debates": total,
        "max_g": round(max_g, 4),
        "avg_g": round(avg_g, 4),
        "max_agent_share": round(max_agent_share, 4),
        "neg_g_count": neg_g,
        "n_judged": len(judge_debates),
    }

epr_seeds = {
    "lot1": [12097, 194647, 233297, 305999, 394699],
    "lot2": [139987, 252079, 291077, 759223, 869321],
    "lot3": [504521, 507919, 555743, 597307, 632813],
    "lot4": [113497, 114967, 160579, 656939, 719821],
}

epr_map = {}
for lot, seeds in epr_seeds.items():
    for s in seeds:
        p = f"experiment_results/{lot}/logs_epr_seed{s}.json"
        if os.path.exists(p):
            epr_map[s] = extract_metrics(load_logs(p))

nodiv_map = {}
for f in sorted(glob.glob("ablation_no_div/logs_epr_no_div_seed*.json")):
    s = int(os.path.basename(f).replace("logs_epr_no_div_seed","").replace(".json",""))
    nodiv_map[s] = extract_metrics(load_logs(f))

judge_map = {}
for f in sorted(glob.glob("ablation_llm_judge/logs_epr_llm_judge_seed*.json")):
    s = int(os.path.basename(f).replace("logs_epr_llm_judge_seed","").replace(".json",""))
    judge_map[s] = extract_metrics(load_logs(f))

common3 = sorted(set(epr_map.keys()) & set(nodiv_map.keys()) & set(judge_map.keys()))

print(f"Seeds with all 3 conditions: {len(common3)}")
print()
print(f"{'seed':>8} | {'cond':>12} | {'deb':>4} | {'max_g':>7} | {'avg_g':>7} | {'max_agt':>7} | {'neg_g':>5} | {'judged':>6}")
print("-" * 80)

for s in common3:
    for label, m in [("EPR", epr_map[s]), ("NO_DIV", nodiv_map[s]), ("LLM_JUDGE", judge_map[s])]:
        print(f"{s:>8} | {label:>12} | {m['n_debates']:>4} | {m['max_g']:>7} | {m['avg_g']:>7} | {m['max_agent_share']:>7} | {m['neg_g_count']:>5} | {m['n_judged']:>6}")
    print()

print("=== AVERAGES across paired seeds ===")
print(f"{'cond':>12} | {'deb':>6} | {'max_g':>7} | {'avg_g':>7} | {'max_agt':>7} | {'neg_g':>5}")
print("-" * 60)
for label, mmap in [("EPR", epr_map), ("NO_DIV", nodiv_map), ("LLM_JUDGE", judge_map)]:
    vals = [mmap[s] for s in common3 if s in mmap]
    if vals:
        print(f"{label:>12} | {sum(v['n_debates'] for v in vals)/len(vals):>6.1f} | {sum(v['max_g'] for v in vals)/len(vals):>7.4f} | {sum(v['avg_g'] for v in vals)/len(vals):>7.4f} | {sum(v['max_agent_share'] for v in vals)/len(vals):>7.4f} | {sum(v['neg_g_count'] for v in vals)/len(vals):>5.1f}")

all_common = sorted(set(epr_map.keys()) & set(nodiv_map.keys()))
print(f"\n=== EPR vs NO_DIV (all {len(all_common)} seeds) ===")
for label, mmap in [("EPR", epr_map), ("NO_DIV", nodiv_map)]:
    vals = [mmap[s] for s in all_common]
    print(f"  {label:>8}: mean max_g={sum(v['max_g'] for v in vals)/len(vals):.4f}  mean avg_g={sum(v['avg_g'] for v in vals)/len(vals):.4f}  mean max_agent_share={sum(v['max_agent_share'] for v in vals)/len(vals):.4f}  mean debates={sum(v['n_debates'] for v in vals)/len(vals):.1f}")
