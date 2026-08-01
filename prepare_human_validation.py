#!/usr/bin/env python3
"""
Prepare human validation subset from benchmark logs.

Reads all logs_{mode}_seed{seed}.json files from a benchmark output directory,
extracts DEBATE claims, performs stratified sampling (100 claims balanced by
condition and temporal window), and generates:

  1. human_validation_subset.csv  — claims ready for annotators
  2. rubrica_anotadores.md        — scoring rubric with anchor definitions

Usage:
  python prepare_human_validation.py --output-dir benchmark_output [--n 100] [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# Agent role labels (from simulation.py AGENT_ROLES)
# ──────────────────────────────────────────────────────────────────────────────

AGENT_ROLE_LABELS: dict[str, str] = {
    "GOV-S1": "Gobierno · S1 Operaciones",
    "GOV-S2": "Gobierno · S2 Coordinación",
    "GOV-S3": "Gobierno · S3 Control",
    "GOV-S4": "Gobierno · S4 Inteligencia",
    "GOV-S5": "Gobierno · S5 Estrategia",
    "OPP-S1": "Oposición · S1 Operaciones",
    "OPP-S2": "Oposición · S2 Coordinación",
    "OPP-S3": "Oposición · S3 Control",
    "OPP-S4": "Oposición · S4 Inteligencia",
    "OPP-S5": "Oposición · S5 Estrategia",
}

# Condition display names
CONDITION_LABELS: dict[str, str] = {
    "epr": "EPR",
    "epr_sham": "EPR-SHAM",
    "langgraph": "LangGraph",
    "epr_q": "EPR-Q",
    "hrrl_no_q": "EPR",
    "baseline": "Baseline",
    "random": "Random",
}

# Conditions to include in validation (EPR vs LangGraph comparison)
VALIDATION_CONDITIONS = {"epr", "hrrl_no_q", "epr_q", "langgraph"}


def load_logs(output_dir: Path) -> list[dict]:
    """Load all log files and return flat list of entries with metadata."""
    all_entries: list[dict] = []
    log_files = sorted(output_dir.glob("logs_*_seed*.json"))

    if not log_files:
        print(f"No log files found in {output_dir}")
        print("Expected files matching: logs_*_seed*.json")
        return all_entries

    for log_path in log_files:
        # Parse mode and seed from filename: logs_{mode}_seed{seed}.json
        stem = log_path.stem  # e.g. logs_epr_seed7
        parts = stem.replace("logs_", "").split("_seed")
        if len(parts) != 2:
            continue
        mode = parts[0]
        seed = int(parts[1])

        with open(log_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            entry["_mode"] = mode
            entry["_seed"] = seed
            all_entries.append(entry)

    return all_entries


def build_node_lookup(entries: list[dict]) -> dict[str, dict]:
    """Build node_id → {claim, agent_id, tick} lookup from log entries."""
    lookup: dict[str, dict] = {}
    for e in entries:
        if e.get("action") == "DEBATE" and e.get("node_id") and e.get("claim"):
            lookup[e["node_id"]] = {
                "claim": e["claim"],
                "agent_id": e.get("agent_id", ""),
                "tick": e.get("tick", 0),
            }
    return lookup


def extract_debate_claims(
    entries: list[dict],
    node_lookup: dict[str, dict],
) -> list[dict]:
    """Extract DEBATE claims with target text resolved."""
    claims: list[dict] = []
    for e in entries:
        if e.get("action") != "DEBATE" or not e.get("claim"):
            continue
        mode = e["_mode"]
        if mode not in VALIDATION_CONDITIONS:
            continue

        target_nid = e.get("target_node_id")
        target_claim = ""
        target_agent = ""
        if target_nid and target_nid in node_lookup:
            target_claim = node_lookup[target_nid]["claim"]
            target_agent = node_lookup[target_nid]["agent_id"]

        tick = e.get("tick", 0)
        max_tick = max((x.get("tick", 0) for x in entries), default=80)
        temporal_window = "temprana" if tick < max_tick / 2 else "tardía"

        claims.append({
            "claim_id": e["node_id"],
            "condition": CONDITION_LABELS.get(mode, mode),
            "condition_raw": mode,
            "seed": e["_seed"],
            "tick": tick,
            "temporal_window": temporal_window,
            "agent_id": e.get("agent_id", ""),
            "agent_role": AGENT_ROLE_LABELS.get(e.get("agent_id", ""), e.get("agent_id", "")),
            "vsm_system": e.get("vsm_system", ""),
            "claim_text": e["claim"],
            "target_node_id": target_nid or "",
            "target_claim_text": target_claim,
            "target_agent_id": target_agent,
            "attack_type": e.get("attack_type", ""),
            "delta_phi": e.get("delta_phi", 0.0),
        })

    return claims


def stratified_sample(claims: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified sample: balance by condition (~50/50 EPR vs LG) and temporal window."""
    rng = random.Random(seed)

    # Group by (condition_group, temporal_window)
    # condition_group: "EPR" for epr/hrrl_no_q, "LG" for langgraph
    strata: dict[str, list[dict]] = defaultdict(list)
    for c in claims:
        group = "EPR" if c["condition_raw"] in ("epr", "hrrl_no_q") else "LG"
        key = f"{group}_{c['temporal_window']}"
        strata[key].append(c)

    # Target: n/4 per stratum (2 groups × 2 windows)
    per_stratum = max(1, n // len(strata)) if strata else n
    sampled: list[dict] = []

    for key, items in strata.items():
        rng.shuffle(items)
        take = min(per_stratum, len(items))
        sampled.extend(items[:take])

    # If under n, fill from remaining
    if len(sampled) < n:
        remaining = [c for c in claims if c not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[: n - len(sampled)])

    # If over n, trim randomly
    if len(sampled) > n:
        rng.shuffle(sampled)
        sampled = sampled[:n]

    # Sort by condition then tick for readability
    sampled.sort(key=lambda c: (c["condition"], c["tick"]))
    return sampled


def write_csv(sampled: list[dict], output_path: Path) -> None:
    """Write CSV with annotator columns."""
    fieldnames = [
        "claim_id",
        "condition",
        "temporal_window",
        "tick",
        "agent_role",
        "claim_text",
        "target_claim_text",
        "attack_type",
        # Annotator 1
        "A1_calidad", "A1_engagement", "A1_novedad",
        # Annotator 2
        "A2_calidad", "A2_engagement", "A2_novedad",
        # Annotator 3
        "A3_calidad", "A3_engagement", "A3_novedad",
        # Notes
        "notas",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in sampled:
            writer.writerow({
                "claim_id": c["claim_id"],
                "condition": c["condition"],
                "temporal_window": c["temporal_window"],
                "tick": c["tick"],
                "agent_role": c["agent_role"],
                "claim_text": c["claim_text"],
                "target_claim_text": c["target_claim_text"] or "(argumento raíz — sin target)",
                "attack_type": c["attack_type"] or "—",
                "A1_calidad": "", "A1_engagement": "", "A1_novedad": "",
                "A2_calidad": "", "A2_engagement": "", "A2_novedad": "",
                "A3_calidad": "", "A3_engagement": "", "A3_novedad": "",
                "notas": "",
            })

    print(f"CSV written: {output_path} ({len(sampled)} claims)")


def write_rubrica(output_path: Path) -> None:
    """Write scoring rubric for annotators."""
    rubrica = """# Rúbrica de anotación — Validación humana

## Instrucciones generales

Van a recibir 100 claims (argumentos) extraídos de un debate multi-agente.
Cada claim fue producido por un agente con un rol específico (Gobierno/Oposición, S1-S5).
Algunos claims atacan a un argumento previo (target); otros son argumentos raíz.

**Importante:** No conocen la condición experimental ni el seed. Evalúen cada claim
por su mérito propio, sin asumir de qué sistema proviene.

## Escala

Cada dimensión se evalúa en una escala de **1 a 5** (enteros):

### 1. Calidad argumentativa

| Puntaje | Descripción |
|---------|-------------|
| 5 | Argumento lógicamente riguroso, con evidencia o razonamiento sólido. Difícil de refutar. |
| 4 | Argumento claro y bien estructurado. Razonamiento válido con minor gaps. |
| 3 | Argumento comprensible pero con debilidades lógicas o falta de evidencia. |
| 2 | Argumento confuso, mal estructurado, o basado en premisas cuestionables. |
| 1 | Incomprensible, irrelevante, o sin sustancia argumentativa. |

### 2. Engagement

¿El claim aborda directamente un argumento oponente específico?

| Puntaje | Descripción |
|---------|-------------|
| 5 | Ataca directamente un punto específico del argumento target, citando o parafraseando. |
| 4 | Se refiere claramente al target pero sin precisión total sobre qué punto ataca. |
| 3 | Mención genérica al target o a la posición oponente. |
| 2 | Referencia vaga o tangencial al debate previo. |
| 1 | No referencia ningún argumento previo. Monólogo desconectado. |

**Nota:** Si el claim es un argumento raíz (sin target), evaluar engagement como
la medida en que establece una posición clara que invita respuesta.

### 3. Novedad

¿El claim introduce una perspectiva nueva o repite argumentos existentes?

| Puntaje | Descripción |
|---------|-------------|
| 5 | Perspectiva completamente nueva no vista en el debate hasta ese punto. |
| 4 | Nuevo ángulo o evidencia sobre un tema existente. |
| 3 | Reformulación de un argumento previo con leve variación. |
| 2 | Repetición de un argumento previo con diferente wording. |
| 1 | Repetición literal o casi literal de un argumento previo. |

## Procedimiento

1. Cada anotador completa las columnas `A1_*`, `A2_*`, o `A3_*` según su número.
2. Anotar **solo enteros del 1 al 5**. Sin decimales.
3. Si un claim es incomprensible o está corrupto (texto truncado, JSON residual),
   anotar 1 en todas las dimensiones y escribir "corrupto" en `notas`.
4. Completar de forma independiente. No discutir puntajes hasta terminar todas las anotaciones.
5. Si hay desacuerdos sistemáticos (>2 puntos de diferencia en >20% de los claims),
   se hará una sesión de recalibración.

## Concordancia

- **Fleiss' κ** entre los 3 anotadores se calculará post-hoc.
- Si κ < 0,6, sesión de recalibración + re-anotación de los claims discrepantes.
- **Spearman ρ** entre el promedio humano y cada juez LLM para validar calibración.
- Umbral de calibración adecuada: ρ > 0,7.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rubrica)
    print(f"Rúbrica written: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare human validation subset from benchmark logs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing logs_{mode}_seed{seed}.json files",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of claims to sample (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="human_validation",
        help="Output file prefix (default: human_validation)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist")
        return

    # 1. Load all logs
    print(f"Loading logs from {output_dir}...")
    entries = load_logs(output_dir)
    if not entries:
        print("No entries found. Exiting.")
        return
    print(f"  Loaded {len(entries)} log entries")

    # 2. Build node_id → claim lookup (to resolve target text)
    node_lookup = build_node_lookup(entries)
    print(f"  Built node lookup: {len(node_lookup)} nodes")

    # 3. Extract DEBATE claims
    claims = extract_debate_claims(entries, node_lookup)
    print(f"  Extracted {len(claims)} DEBATE claims")

    if len(claims) < args.n:
        print(f"  Warning: only {len(claims)} claims available, sampling all")

    # 4. Stratified sample
    sampled = stratified_sample(claims, args.n, args.seed)
    print(f"  Sampled {len(sampled)} claims (stratified by condition × temporal window)")

    # Print stratum distribution
    from collections import Counter
    dist = Counter(f"{c['condition']}_{c['temporal_window']}" for c in sampled)
    for k, v in sorted(dist.items()):
        print(f"    {k}: {v}")

    # 5. Write CSV
    csv_path = output_dir / f"{args.prefix}_subset.csv"
    write_csv(sampled, csv_path)

    # 6. Write rúbrica
    rubrica_path = output_dir / "rubrica_anotadores.md"
    write_rubrica(rubrica_path)

    print(f"\nDone. Send {csv_path} to 3 annotators along with {rubrica_path}.")


if __name__ == "__main__":
    main()
