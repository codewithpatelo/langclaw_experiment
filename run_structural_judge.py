"""Blind LLM assessment of structural debate quality, window by window.

Why
---
The per-argument judging campaign scores each claim in isolation, so it cannot
speak to properties that only exist at the level of the debate as a whole:
whether positions stay diverse, whether agents genuinely engage with each other,
whether the exchange turns repetitive, whether participation stays balanced.
Those are exactly the properties the internal signal g was supposed to capture,
and the claim that g tracks them has never been tested against an external
measure.

This script asks two LLM judges to evaluate the argument subgraph of each
temporal window as a unit. Because the same windows are used as in the
per-argument analysis, the result answers directly whether structural quality
degrades over the course of a debate, and whether the activation regime makes a
difference.

Blinding
--------
Judges never see the mode. Agent identifiers are replaced by stable anonymous
labels within each run, and node identifiers by short ordinals, so that naming
conventions cannot leak the condition.

Cost
----
4 modes x 20 seeds x 5 windows x 2 judges = 800 calls. Checkpointed per
(mode, judge) and resumable.

Usage
-----
    python run_structural_judge.py
    python run_structural_judge.py --concurrency 16
    python run_structural_judge.py --clean
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from run_judge_smoke import (
    KeyRotator,
    get_completed_seeds,
    load_env,
    load_logs_for_seed,
)

BASE_DIR = Path(__file__).parent
MODES = ["epr", "epr_q", "epr_sham", "langgraph"]
LOTS = ["lot1", "lot2", "lot3", "lot4"]
TOTAL_TICKS = 80
N_WINDOWS = 5
CHECKPOINT_PATH = BASE_DIR / "structural_judge_checkpoint.json"
RESULTS_PATH = BASE_DIR / "structural_judge_results.json"
DEFAULT_CONCURRENCY = 12
CHECKPOINT_EVERY = 5
JUDGE_SEED = 42

_STRUCTURAL_PROMPT = """Eres un evaluador experto en calidad de deliberacion argumentativa. Se te presenta un fragmento de un debate multiagente sobre el siguiente tema:

TEMA: {topic}

A continuacion, los argumentos producidos durante un intervalo del debate. Cada argumento indica su autor (etiqueta anonima), el argumento al que responde, y el tipo de ataque dialectico.

{graph}

Evalua el fragmento COMO CONJUNTO, no argumento por argumento. Puntua cada dimension de 1 a 10:

1. diversidad: variedad de posiciones y perspectivas sustantivas presentes. 10 = multiples posiciones distintas y bien diferenciadas; 1 = todos sostienen lo mismo.

2. enganche: profundidad con que los argumentos responden efectivamente a aquello que atacan. 10 = las objeciones abordan el contenido especifico del objetivo; 1 = los argumentos se ignoran mutuamente o responden de forma generica.

3. no_redundancia: ausencia de repeticion. 10 = cada argumento aporta contenido nuevo; 1 = los agentes reiteran las mismas ideas con otras palabras.

4. balance: distribucion de la participacion. 10 = los agentes contribuyen de forma equilibrada; 1 = uno o dos agentes monopolizan el intercambio.

5. global: calidad estructural del fragmento como deliberacion. No es el promedio de las anteriores: es tu juicio integrado sobre si este fragmento constituye un buen debate.

Responde UNICAMENTE con JSON valido:
{{"diversidad": <1-10>, "enganche": <1-10>, "no_redundancia": <1-10>, "balance": <1-10>, "global": <1-10>, "justificacion": "<una frase>"}}"""


def window_of(tick: int) -> int:
    width = TOTAL_TICKS / N_WINDOWS
    return min(int(tick / width), N_WINDOWS - 1)


def serialise_window(debates: list, window: int) -> tuple[str, int]:
    """Render the argument subgraph of one window with anonymised identifiers.

    Anonymisation is computed over the whole run, not the window, so that a
    given agent keeps the same label across windows of the same debate.
    """
    agent_labels: dict[str, str] = {}
    for d in debates:
        if d.agent_id not in agent_labels:
            agent_labels[d.agent_id] = f"A{len(agent_labels) + 1}"

    node_labels: dict[str, str] = {}
    claims_by_node: dict[str, str] = {}
    agents_by_node: dict[str, str] = {}
    for i, d in enumerate(debates, start=1):
        if d.node_id:
            node_labels[d.node_id] = f"#{i}"
            claims_by_node[d.node_id] = d.claim or ""
            agents_by_node[d.node_id] = agent_labels.get(d.agent_id, "A?")

    in_window = [d for d in debates if window_of(d.tick) == window]
    if not in_window:
        return "", 0

    lines: list[str] = []
    for d in in_window:
        label = node_labels.get(d.node_id, "#?")
        author = agent_labels.get(d.agent_id, "A?")
        if d.target_node_id and d.target_node_id in claims_by_node:
            target_label = node_labels.get(d.target_node_id, "#?")
            target_author = agents_by_node.get(d.target_node_id, "A?")
            target_claim = claims_by_node[d.target_node_id][:200]
            head = (
                f"[{label}] {author} responde a {target_label} "
                f"({d.attack_type or 'sin tipo'})"
            )
            body = f'    Argumento: "{(d.claim or "")[:400]}"'
            tgt = f'    Objetivo {target_label} ({target_author}): "{target_claim}"'
            lines.append("\n".join([head, body, tgt]))
        else:
            head = f"[{label}] {author} (argumento raiz)"
            body = f'    Argumento: "{(d.claim or "")[:400]}"'
            lines.append("\n".join([head, body]))

    return "\n\n".join(lines), len(in_window)


def build_units() -> list[dict[str, Any]]:
    """One evaluation unit per (mode, seed, window) that contains arguments."""
    from langclaw.agent import DEBATE_TOPIC

    units: list[dict[str, Any]] = []
    for lot in LOTS:
        for seed in get_completed_seeds(lot):
            for mode in MODES:
                logs = load_logs_for_seed(lot, mode, seed)
                if not logs:
                    continue
                debates = [l for l in logs if l.action == "DEBATE" and l.claim]
                if not debates:
                    continue
                for w in range(N_WINDOWS):
                    graph, n_args = serialise_window(debates, w)
                    if n_args < 3:
                        continue
                    units.append({
                        "unit_id": f"{mode}|{lot}|{seed}|w{w}",
                        "mode": mode,
                        "lot": lot,
                        "seed": seed,
                        "window": w,
                        "n_arguments": n_args,
                        "prompt": _STRUCTURAL_PROMPT.format(
                            topic=DEBATE_TOPIC, graph=graph
                        ),
                    })
    return units


def judge_unit(
    unit: dict, *, model: str, base_url: str, key_rotator: KeyRotator
) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=key_rotator.next())
    keys = ["diversidad", "enganche", "no_redundancia", "balance", "global"]
    try:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": unit["prompt"]}],
            "max_completion_tokens": 500,
            "temperature": 0.0,
            "seed": JUDGE_SEED,
        }
        if "glm" in model.lower():
            kwargs["extra_body"] = {"do_sample": False}
        response = client.chat.completions.create(**kwargs)
        raw = (response.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {"unit_id": unit["unit_id"], "scores": {}, "error": raw[:200]}
        parsed = json.loads(match.group())
        scores = {}
        for k in keys:
            try:
                scores[k] = max(1, min(10, int(parsed[k])))
            except (KeyError, TypeError, ValueError):
                scores[k] = 0
        return {
            "unit_id": unit["unit_id"],
            "scores": scores,
            "justificacion": str(parsed.get("justificacion", ""))[:300],
        }
    except Exception as exc:
        return {"unit_id": unit["unit_id"], "scores": {}, "error": f"{exc}"[:200]}


_checkpoint_lock = threading.Lock()


def load_checkpoint(clean: bool) -> dict:
    if clean or not CHECKPOINT_PATH.exists():
        return {}
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print("[WARN] checkpoint corrupted, starting fresh")
        return {}


def save_checkpoint(data: dict) -> None:
    with _checkpoint_lock:
        tmp = CHECKPOINT_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(CHECKPOINT_PATH)


def run_judge(
    units: list[dict],
    *,
    model: str,
    base_url: str,
    key_rotator: KeyRotator,
    checkpoint: dict,
    concurrency: int,
) -> dict[str, dict]:
    done: dict[str, dict] = dict(checkpoint.get(model, {}))
    pending = [u for u in units if u["unit_id"] not in done]
    print(f"    [{model}] {len(done)} ya evaluadas, {len(pending)} pendientes")
    if not pending:
        return done

    start = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                judge_unit, u, model=model, base_url=base_url, key_rotator=key_rotator
            ): u
            for u in pending
        }
        for fut in as_completed(futures):
            res = fut.result()
            done[res["unit_id"]] = res
            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                checkpoint[model] = done
                save_checkpoint(checkpoint)
                rate = completed / max(1e-9, time.time() - start)
                eta = (len(pending) - completed) / max(1e-9, rate)
                print(
                    f"    [{model}] {completed}/{len(pending)} "
                    f"({rate:.2f}/s, ETA {eta:.0f}s)"
                )

    checkpoint[model] = done
    save_checkpoint(checkpoint)
    print(f"    [{model}] completado en {time.time() - start:.0f}s")
    return done


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blind structural judging of debate windows"
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    env = load_env()
    deepseek_keys = [
        env.get(k, "")
        for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_2", "DEEPSEEK_API_KEY_3", "DEEPSEEK_API_KEY_4")
    ]
    zai_keys = [
        env.get(k, "")
        for k in ("ZAI_API_KEY", "ZAI_API_KEY_2", "ZAI_API_KEY_3", "ZAI_API_KEY_4")
    ]
    deepseek_keys = [k for k in deepseek_keys if k]
    zai_keys = [k for k in zai_keys if k]
    if not deepseek_keys or not zai_keys:
        print("ERROR: faltan claves de API")
        return 1

    judges = [
        {
            "model": "deepseek-v4-pro",
            "base_url": "https://api.deepseek.com/v1",
            "key_rotator": KeyRotator(deepseek_keys),
        },
        {
            "model": "glm-5.2",
            "base_url": "https://api.z.ai/api/paas/v4/",
            "key_rotator": KeyRotator(zai_keys),
        },
    ]

    print("Construyendo unidades de evaluacion (modo, semilla, ventana)...")
    units = build_units()
    by_mode: dict[str, int] = {}
    for u in units:
        by_mode[u["mode"]] = by_mode.get(u["mode"], 0) + 1
    print(f"{len(units)} unidades: " + ", ".join(f"{m}={n}" for m, n in by_mode.items()))
    print(f"Llamadas totales: {len(units) * len(judges)}")

    checkpoint = load_checkpoint(args.clean)
    if checkpoint:
        print(f"[RESUME] checkpoint con {len(checkpoint)} jueces")

    print("\nEvaluando con ambos jueces en paralelo...")
    results_by_judge: dict[str, dict[str, dict]] = {}
    with ThreadPoolExecutor(max_workers=len(judges)) as pool:
        futures = {
            pool.submit(
                run_judge,
                units,
                model=j["model"],
                base_url=j["base_url"],
                key_rotator=j["key_rotator"],
                checkpoint=checkpoint,
                concurrency=args.concurrency,
            ): j["model"]
            for j in judges
        }
        for fut, model in futures.items():
            results_by_judge[model] = fut.result()

    payload = {
        "units": {
            u["unit_id"]: {
                "mode": u["mode"],
                "seed": u["seed"],
                "window": u["window"],
                "n_arguments": u["n_arguments"],
            }
            for u in units
        },
        "judges": results_by_judge,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    valid = {
        m: sum(1 for r in res.values() if r.get("scores", {}).get("global", 0) > 0)
        for m, res in results_by_judge.items()
    }
    print("\nEvaluaciones validas por juez: " + ", ".join(f"{m}={n}" for m, n in valid.items()))
    print(f"Resultados en: {RESULTS_PATH}")
    print("\nAnaliza con: python analyze_structural_judge.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
