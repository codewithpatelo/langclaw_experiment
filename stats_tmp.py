import json, glob, os
from scipy import stats
import numpy as np

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
    
    return {
        "n_debates": total,
        "max_g": max(g_values) if g_values else 0,
        "avg_g": sum(g_values) / len(g_values) if g_values else 0,
        "max_agent_share": max_agent_share,
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

paired = sorted(set(epr_map.keys()) & set(nodiv_map.keys()))
print(f"Paired seeds: {len(paired)}")
print(f"Seeds: {paired}")
print()

# Verify data integrity
print("=== Data integrity ===")
for s in paired:
    lot_key = [k for k,v in epr_seeds.items() if s in v][0]
    epr_logs = load_logs(f"experiment_results/{lot_key}/logs_epr_seed{s}.json")
    nodiv_path = f"ablation_no_div/logs_epr_no_div_seed{s}.json"
    nodiv_logs = load_logs(nodiv_path)
    epr_max_tick = max(l.get("tick", 0) for l in epr_logs)
    nodiv_max_tick = max(l.get("tick", 0) for l in nodiv_logs)
    epr_debates = sum(1 for l in epr_logs if l.get("action") == "DEBATE" and l.get("claim"))
    nodiv_debates = sum(1 for l in nodiv_logs if l.get("action") == "DEBATE" and l.get("claim"))
    ok = "OK" if epr_max_tick == 80 and nodiv_max_tick == 80 else "MISMATCH"
    print(f"  seed={s}: EPR tick={epr_max_tick} debates={epr_debates} | NO_DIV tick={nodiv_max_tick} debates={nodiv_debates} [{ok}]")

print()

metrics = ["max_agent_share", "max_g", "avg_g", "n_debates"]
labels = {
    "max_agent_share": "max_share (concentración del discurso)",
    "max_g": "max_g (señal de calidad máxima)",
    "avg_g": "avg_g (señal de calidad media)",
    "n_debates": "n_debates (volumen total)",
}

n_metrics = len(metrics)
alpha = 0.05
bonferroni_alpha = alpha / n_metrics

print("=" * 100)
print(f"Wilcoxon signed-rank test (n={len(paired)}, alpha={alpha}, Bonferroni alpha={bonferroni_alpha:.4f})")
print("=" * 100)
print(f"{'Métrica':<45} | {'EPR':>8} | {'NO_DIV':>8} | {'diff':>8} | {'W':>6} | {'p':>10} | {'p_bonf':>10} | sig")
print("-" * 100)

results = {}
for m in metrics:
    epr_vals = np.array([epr_map[s][m] for s in paired])
    nodiv_vals = np.array([nodiv_map[s][m] for s in paired])
    
    epr_mean = np.mean(epr_vals)
    nodiv_mean = np.mean(nodiv_vals)
    diff = nodiv_mean - epr_mean
    
    w_stat, p_val = stats.wilcoxon(nodiv_vals, epr_vals)
    p_bonf = min(p_val * n_metrics, 1.0)
    
    # Effect size: Cliff's delta
    n_epr = len(epr_vals)
    cliff = 0
    for i in range(n_epr):
        for j in range(n_epr):
            d = nodiv_vals[i] - epr_vals[j]
            if d > 0: cliff += 1
            elif d < 0: cliff -= 1
    cliff_delta = cliff / (n_epr * n_epr)
    
    sig = ""
    if p_bonf < 0.001: sig = "***"
    elif p_bonf < 0.01: sig = "**"
    elif p_bonf < 0.05: sig = "*"
    else: sig = "ns"
    
    results[m] = {
        "epr_mean": epr_mean,
        "nodiv_mean": nodiv_mean,
        "diff": diff,
        "w_stat": w_stat,
        "p_val": p_val,
        "p_bonf": p_bonf,
        "cliff": cliff_delta,
        "sig": sig,
    }
    
    print(f"{labels[m]:<45} | {epr_mean:>8.4f} | {nodiv_mean:>8.4f} | {diff:>+8.4f} | {w_stat:>6.1f} | {p_val:>10.6f} | {p_bonf:>10.6f} | {sig}")

print()
print("=== Detalle max_share por seed ===")
print(f"  {'seed':>8} | {'EPR':>8} | {'NO_DIV':>8} | {'diff':>8} | {'dir':>5}")
for s in paired:
    diff = nodiv_map[s]["max_agent_share"] - epr_map[s]["max_agent_share"]
    direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
    print(f"  {s:>8} | {epr_map[s]['max_agent_share']:>8.4f} | {nodiv_map[s]['max_agent_share']:>8.4f} | {diff:>+8.4f} | {direction:>5}")

r = results["max_agent_share"]
print(f"\n=== Conclusión max_share ===")
print(f"  EPR mean:     {r['epr_mean']:.4f}")
print(f"  NO_DIV mean:  {r['nodiv_mean']:.4f}")
print(f"  Diferencia:   {r['diff']:+.4f}")
print(f"  Wilcoxon W={r['w_stat']:.1f}, p={r['p_val']:.6f}")
print(f"  p Bonferroni: {r['p_bonf']:.6f} ({'significativo' if r['p_bonf'] < 0.05 else 'NO significativo'})")
print(f"  Cliff's delta: {r['cliff']:.4f}")
print(f"  Significativo tras Bonferroni: {'Sí' if r['p_bonf'] < 0.05 else 'NO'}")
print()
if r['p_bonf'] >= 0.05:
    print("  >>> No hay diferencia significativa en max_share tras remover diversidad de g.")
    print("  >>> La equidad del discurso NO depende del término de diversidad en g.")
    print("  >>> Emerige de la interacción entre déficits locales (homeostasis), no del diseño de g.")
else:
    print("  >>> Hay diferencia significativa en max_share tras remover diversidad de g.")
    print("  >>> La equidad del discurso depende parcialmente del término de diversidad en g.")
