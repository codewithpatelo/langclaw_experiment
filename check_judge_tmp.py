import json

# Check no_div checkpoint
d = json.load(open("ablation_no_div/run_checkpoints/epr_no_div__seed12097.json", encoding="utf-8"))
print(f"=== EPR_NO_DIV checkpoint ===")
print(f"tick: {d.get('next_tick')}")
logs = d.get("env", {}).get("logs", [])
debates = [l for l in logs if l.get("action") == "DEBATE"]
print(f"debates so far: {len(debates)}")
print(f"last 5 log entries:")
for l in logs[-5:]:
    print(f"  tick={l['tick']} agent={l['agent_id']} action={l['action']} act_prob={l.get('activation_prob',0):.4f}")

# Check llm_judge checkpoint
d2 = json.load(open("ablation_llm_judge/run_checkpoints/epr_llm_judge__seed12097.json", encoding="utf-8"))
print(f"\n=== EPR_LLM_JUDGE checkpoint ===")
print(f"tick: {d2.get('next_tick')}")
logs2 = d2.get("env", {}).get("logs", [])
debates2 = [l for l in logs2 if l.get("action") == "DEBATE"]
print(f"debates so far: {len(debates2)}")
has_judge = sum(1 for l in debates2 if l.get("judge_info"))
print(f"debates with judge_info: {has_judge}/{len(debates2)}")
if debates2:
    d0 = debates2[0]
    ji = d0.get("judge_info", {})
    print(f"first debate: tick={d0['tick']} g={d0.get('delta_phi','?')} n_flags={ji.get('n_flags','?')} fluidez={ji.get('fluidez','?')}")
