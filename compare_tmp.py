import json, glob, os, statistics

def load_logs(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def extract_metrics(logs):
    debates = [l for l in logs if l.get("action") == "DEBATE" and l.get("claim")]
    g_values = [l.get("delta_phi", 0) for l in debates]
    agent_counts = {}
    for l in debates:
        aid = l.get("agent_id", "?")
        agent_counts[aid] = agent_counts.get(aid, 0) + 1
    total = len(debates)
    max_agent_share = max(agent_counts.values()) / total if total else 0
    neg_g = sum(1 for g in g_values if g < 0)
    return {
        "n_debates": total,
        "max_g": max(g_values) if g_values else 0,
        "avg_g": sum(g_values) / len(g_values) if g_values else 0,
        "max_agent_share": max_agent_share,
        "neg_g_count": neg_g,
    }

# EPR (main experiment)
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

# LangGraph (main experiment)
lg_map = {}
for lot, seeds in epr_seeds.items():
    for s in seeds:
        p = f"experiment_results/{lot}/logs_langgraph_seed{s}.json"
        if os.path.exists(p):
            lg_map[s] = extract_metrics(load_logs(p))

# LLM_JUDGE (all logs, deduplicated by seed)
judge_map = {}
for f in sorted(glob.glob("ablation_llm_judge/**/logs_epr_llm_judge_seed*.json", recursive=True)):
    s = int(os.path.basename(f).replace("logs_epr_llm_judge_seed","").replace(".json",""))
    if s not in judge_map:  # deduplicate
        judge_map[s] = extract_metrics(load_logs(f))

# Paired seeds across all 3 conditions
paired = sorted(set(epr_map.keys()) & set(lg_map.keys()) & set(judge_map.keys()))
print(f"Paired seeds (EPR + LG + LLM_JUDGE): {len(paired)}")
print(f"Seeds: {paired}")
print()

print(f"{'seed':>8} | {'cond':>12} | {'deb':>4} | {'max_g':>7} | {'avg_g':>7} | {'max_agt':>7} | {'neg_g':>5}")
print("-" * 75)
for s in paired:
    for label, m in [("EPR", epr_map[s]), ("LangGraph", lg_map[s]), ("LLM_JUDGE", judge_map[s])]:
        print(f"{s:>8} | {label:>12} | {m['n_debates']:>4} | {m['max_g']:>7.4f} | {m['avg_g']:>7.4f} | {m['max_agent_share']:>7.4f} | {m['neg_g_count']:>5}")
    print()

print("=== AVERAGES ===")
print(f"{'cond':>12} | {'deb':>6} | {'max_g':>7} | {'avg_g':>7} | {'max_agt':>7} | {'neg_g':>5}")
print("-" * 55)
for label, mmap in [("EPR", epr_map), ("LangGraph", lg_map), ("LLM_JUDGE", judge_map)]:
    vals = [mmap[s] for s in paired if s in mmap]
    if vals:
        print(f"{label:>12} | {statistics.mean([v['n_debates'] for v in vals]):>6.1f} | {statistics.mean([v['max_g'] for v in vals]):>7.4f} | {statistics.mean([v['avg_g'] for v in vals]):>7.4f} | {statistics.mean([v['max_agent_share'] for v in vals]):>7.4f} | {statistics.mean([v['neg_g_count'] for v in vals]):>5.1f}")
