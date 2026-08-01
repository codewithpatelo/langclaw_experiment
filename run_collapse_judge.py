"""Detection of context collapse in multi-agent debate, grounded on the transcript.

What context collapse is, and why quality scores cannot detect it
----------------------------------------------------------------
Context collapse is not low argument quality. It is a family of fidelity
failures that appear as a model loses reliable access to earlier context:
content that was said is silently dropped, content that was never said is
reconstructed, positions are assigned to the wrong participant, distinct threads
are merged, and settled points are relitigated as if new.

The defining property is that the output remains fluent and locally coherent
while these errors accumulate. The literature calls this the High-Functioning
Compensation Effect: fluency masks missing information, so higher surface quality
makes the failure harder, not easier, to notice. Session-level analyses make the
methodological consequence explicit -- per-turn quality scores look fine while
context retrieval is already failing.

Two design errors in the earlier judging campaign follow from this:

  1. It scored argument quality (relevance, originality, argumentative force).
     Those are precisely the dimensions that stay intact under collapse.
  2. It showed the judge only five preceding turns, so an argument that
     contradicted or forgot something established forty pulses earlier was
     indistinguishable from a sound one.

This script fixes both. Judges receive the complete prior transcript of the run
and are asked for verifiable fidelity violations against it. Rhetorical quality
is explicitly declared irrelevant, and is scored separately so that the
dissociation between fluency and fidelity can be measured directly.

Operationalised failure modes
-----------------------------
  fabricacion         attributes to the debate a claim, position or datum absent
                      from the record
  misatribucion       assigns a position to an agent who did not hold it
  distorsion_objetivo mischaracterises the argument being responded to
  amnesia             reintroduces a point already made or already answered, as
                      though it were new
  conflacion          merges two distinct prior arguments, or links unrelated
                      ideas as if connected

Each flag must be accompanied by a verbatim quotation. Flags without supporting
evidence are discarded during analysis, which bounds false positives.

Sampling
--------
Two arguments per (mode, seed, window), stratified, giving 800 units and 40
observations per (mode, window) cell once pooled over seeds, while still allowing
per-seed rates for paired inference. Both judges evaluate every unit.

Usage
-----
    python run_collapse_judge.py
    python run_collapse_judge.py --per-cell 2 --concurrency 16
    python run_collapse_judge.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
CHECKPOINT_PATH = BASE_DIR / "collapse_judge_checkpoint.json"
RESULTS_PATH = BASE_DIR / "collapse_judge_results.json"
DEFAULT_CONCURRENCY = 16
DEFAULT_PER_CELL = 2
CHECKPOINT_EVERY = 10
JUDGE_SEED = 42
SAMPLING_SEED = 20260730

FAILURE_MODES = [
    "fabricacion",
    "misatribucion",
    "distorsion_objetivo",
    "amnesia",
    "conflacion",
]

_COLLAPSE_PROMPT = """Eres un auditor de fidelidad de transcripciones. Tu tarea NO es evaluar si un argumento es bueno, persuasivo o bien escrito. Tu unica tarea es verificar si el argumento es FIEL al registro de lo que efectivamente se dijo antes en el debate.

TEMA DEL DEBATE: {topic}

=== REGISTRO COMPLETO DE ARGUMENTOS PREVIOS ===
{history}

=== ARGUMENTO AL QUE RESPONDE ===
{target}

=== ARGUMENTO BAJO AUDITORIA (turno {position} del debate) ===
Autor: {author}
Texto: "{claim}"

=== INSTRUCCIONES ===
Verifica el argumento bajo auditoria contra el registro previo. Detecta unicamente violaciones de fidelidad VERIFICABLES. Marca 1 solo si podes citar textualmente la parte del argumento que constituye la violacion; si no podes citarla, marca 0.

fabricacion: el argumento atribuye al debate una afirmacion, posicion o dato que NO aparece en el registro previo. No cuenta aportar informacion nueva propia; cuenta presentar como ya dicho algo que nunca se dijo.

misatribucion: el argumento asigna una posicion a un agente que no la sostuvo, o confunde quien dijo que.

distorsion_objetivo: el argumento tergiversa el contenido del argumento al que responde, atacando algo distinto de lo que ese argumento efectivamente afirma.

amnesia: el argumento reintroduce un punto ya planteado antes como si fuera nuevo, o ignora que ese punto ya fue respondido o zanjado en el registro.

conflacion: el argumento fusiona dos argumentos previos distintos como si fueran uno, o vincula ideas que en el registro no estaban relacionadas.

Ademas, de forma INDEPENDIENTE de lo anterior, puntua:

fluidez (1-10): que tan bien escrito, coherente y persuasivo suena el argumento leido en aislamiento, sin considerar su fidelidad al registro.

Se conservador: marca una violacion solo con evidencia clara en el texto. Un argumento puede ser perfectamente fiel y merecer todos los flags en 0.

Responde UNICAMENTE con JSON valido:
{{"fabricacion": <0|1>, "misatribucion": <0|1>, "distorsion_objetivo": <0|1>, "amnesia": <0|1>, "conflacion": <0|1>, "fluidez": <1-10>, "evidencia": "<cita textual que justifica los flags marcados, o cadena vacia si todos son 0>"}}"""


def window_of(tick: int) -> int:
    return min(int(tick / (TOTAL_TICKS / N_WINDOWS)), N_WINDOWS - 1)


def build_run_units(
    logs: list, mode: str, lot: str, seed: int, rng: random.Random, per_cell: int
) -> list[dict[str, Any]]:
    """Sample `per_cell` arguments from each temporal window of one run.

    The prompt for each sampled argument carries the entire preceding transcript,
    so that omissions and fabrications relative to distant context are visible.
    Identifiers are anonymised with labels stable within the run.
    """
    from langclaw.agent import DEBATE_TOPIC

    debates = [l for l in logs if l.action == "DEBATE" and l.claim]
    if len(debates) < 6:
        return []

    agent_labels: dict[str, str] = {}
    for d in debates:
        if d.agent_id not in agent_labels:
            agent_labels[d.agent_id] = f"A{len(agent_labels) + 1}"

    node_labels: dict[str, str] = {}
    claims_by_node: dict[str, str] = {}
    for i, d in enumerate(debates, start=1):
        if d.node_id:
            node_labels[d.node_id] = f"#{i}"
            claims_by_node[d.node_id] = d.claim or ""

    by_window: dict[int, list[int]] = {}
    for i, d in enumerate(debates):
        by_window.setdefault(window_of(d.tick), []).append(i)

    units: list[dict[str, Any]] = []
    for w in range(N_WINDOWS):
        candidates = [i for i in by_window.get(w, []) if i >= 3]
        if not candidates:
            continue
        chosen = rng.sample(candidates, min(per_cell, len(candidates)))
        for idx in chosen:
            d = debates[idx]
            history_lines = []
            for j in range(idx):
                prev = debates[j]
                plabel = node_labels.get(prev.node_id, f"#{j+1}")
                pauthor = agent_labels.get(prev.agent_id, "A?")
                ptarget = (
                    node_labels.get(prev.target_node_id, "raiz")
                    if prev.target_node_id
                    else "raiz"
                )
                history_lines.append(
                    f'{plabel} {pauthor} (pulso {prev.tick}, responde a {ptarget}): '
                    f'"{(prev.claim or "")[:260]}"'
                )
            history = "\n".join(history_lines) if history_lines else "Sin argumentos previos."

            if d.target_node_id and d.target_node_id in claims_by_node:
                tlabel = node_labels.get(d.target_node_id, "#?")
                target = f'{tlabel}: "{claims_by_node[d.target_node_id][:400]}"'
            else:
                target = "Ninguno (argumento raiz)."

            units.append({
                "unit_id": f"{mode}|{lot}|{seed}|w{w}|{idx}",
                "mode": mode,
                "lot": lot,
                "seed": seed,
                "window": w,
                "tick": d.tick,
                "position": idx + 1,
                "n_prior": idx,
                "node_id": d.node_id,
                "prompt": _COLLAPSE_PROMPT.format(
                    topic=DEBATE_TOPIC,
                    history=history,
                    target=target,
                    position=idx + 1,
                    author=agent_labels.get(d.agent_id, "A?"),
                    claim=(d.claim or "")[:800],
                ),
            })
    return units


def build_units(per_cell: int) -> list[dict[str, Any]]:
    rng = random.Random(SAMPLING_SEED)
    units: list[dict[str, Any]] = []
    for lot in LOTS:
        for seed in get_completed_seeds(lot):
            for mode in MODES:
                logs = load_logs_for_seed(lot, mode, seed)
                if logs:
                    units.extend(build_run_units(logs, mode, lot, seed, rng, per_cell))
    return units


MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_S = 120.0


def audit_unit(
    unit: dict, *, model: str, base_url: str, key_rotator: KeyRotator
) -> dict:
    """Audit one argument, retrying transient failures with exponential backoff.

    A fresh key is drawn on every attempt, so a unit that hits a per-key rate
    limit is likely to land on a different key when retried.
    """
    from openai import OpenAI

    last_error = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            client = OpenAI(
                base_url=base_url,
                api_key=key_rotator.next(),
                timeout=REQUEST_TIMEOUT_S,
                max_retries=0,
            )
            kwargs: dict = {
                "model": model,
                "messages": [{"role": "user", "content": unit["prompt"]}],
                "max_completion_tokens": 600,
                "temperature": 0.0,
                "seed": JUDGE_SEED,
            }
            if "glm" in model.lower():
                kwargs["extra_body"] = {"do_sample": False}
            response = client.chat.completions.create(**kwargs)
            raw = (response.choices[0].message.content or "").strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                last_error = f"unparseable: {raw[:150]}"
                time.sleep(2 ** attempt)
                continue

            parsed = json.loads(match.group())
            flags = {}
            for m in FAILURE_MODES:
                try:
                    flags[m] = 1 if int(parsed.get(m, 0)) == 1 else 0
                except (TypeError, ValueError):
                    flags[m] = 0
            try:
                fluidez = max(1, min(10, int(parsed.get("fluidez", 0))))
            except (TypeError, ValueError):
                fluidez = 0
            return {
                "unit_id": unit["unit_id"],
                "ok": True,
                "flags": flags,
                "fluidez": fluidez,
                "evidencia": str(parsed.get("evidencia", ""))[:400],
                "attempts": attempt + 1,
            }
        except Exception as exc:
            last_error = f"{exc}"[:200]
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 ** attempt)

    return {"unit_id": unit["unit_id"], "ok": False, "error": last_error}


_lock = threading.Lock()


def save_checkpoint(data: dict) -> None:
    """Persist the checkpoint atomically and safely under concurrent judges.

    Both judges share the outer dict, each owning one nested dict that it keeps
    mutating as results arrive. Serialising the shared structure directly lets
    one judge's writer iterate the other judge's dict mid-mutation, which raises
    "dictionary changed size during iteration" and kills the offending thread.
    A shallow snapshot of each nested mapping is taken under the lock so that
    json never walks a live dictionary.
    """
    with _lock:
        snapshot = {
            judge: dict(results) if isinstance(results, dict) else results
            for judge, results in data.items()
        }
        tmp = CHECKPOINT_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
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
    # A stored failure carries no rating, so it must not count as completed;
    # otherwise a transient rate-limit error would permanently drop the unit.
    succeeded = {uid for uid, r in done.items() if r.get("ok")}
    pending = [u for u in units if u["unit_id"] not in succeeded]
    n_retry = len(done) - len(succeeded)
    print(
        f"    [{model}] {len(succeeded)} listas, {len(pending)} pendientes"
        + (f" ({n_retry} reintentos de fallos previos)" if n_retry else "")
    )
    if not pending:
        return done

    start = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(audit_unit, u, model=model, base_url=base_url, key_rotator=key_rotator)
            for u in pending
        ]
        for fut in as_completed(futures):
            res = fut.result()
            done[res["unit_id"]] = res
            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                checkpoint[model] = done
                save_checkpoint(checkpoint)
                rate = completed / max(1e-9, time.time() - start)
                print(
                    f"    [{model}] {completed}/{len(pending)} "
                    f"({rate:.2f}/s, ETA {(len(pending)-completed)/max(1e-9,rate):.0f}s)"
                )

    checkpoint[model] = done
    save_checkpoint(checkpoint)
    print(f"    [{model}] completado en {time.time()-start:.0f}s")
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit context collapse against the transcript")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--per-cell", type=int, default=DEFAULT_PER_CELL)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="build units and show one prompt")
    args = parser.parse_args()

    print("Construyendo unidades de auditoria...")
    units = build_units(args.per_cell)
    by_mode: dict[str, int] = {}
    by_window: dict[int, int] = {}
    for u in units:
        by_mode[u["mode"]] = by_mode.get(u["mode"], 0) + 1
        by_window[u["window"]] = by_window.get(u["window"], 0) + 1
    lengths = [len(u["prompt"]) for u in units]
    print(f"{len(units)} unidades | por modo: {by_mode}")
    print(f"por ventana: {dict(sorted(by_window.items()))}")
    print(f"prompt chars: min={min(lengths)} media={sum(lengths)//len(lengths)} max={max(lengths)}")
    print(f"llamadas totales: {len(units)*2}")

    if args.dry_run:
        sample = next(u for u in units if u["window"] == 4)
        print("\n" + "=" * 70)
        print(f"MUESTRA {sample['unit_id']} — {sample['n_prior']} argumentos previos")
        print("=" * 70)
        p = sample["prompt"]
        print(p[:1500])
        print("\n[... registro intermedio omitido ...]\n")
        print(p[-1800:])
        return 0

    env = load_env()
    ds = [env.get(k, "") for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_2", "DEEPSEEK_API_KEY_3", "DEEPSEEK_API_KEY_4")]
    za = [env.get(k, "") for k in ("ZAI_API_KEY", "ZAI_API_KEY_2", "ZAI_API_KEY_3", "ZAI_API_KEY_4")]
    ds = [k for k in ds if k]
    za = [k for k in za if k]
    if not ds or not za:
        print("ERROR: faltan claves de API")
        return 1

    judges = [
        {"model": "deepseek-v4-pro", "base_url": "https://api.deepseek.com/v1", "key_rotator": KeyRotator(ds)},
        {"model": "glm-5.2", "base_url": "https://api.z.ai/api/paas/v4/", "key_rotator": KeyRotator(za)},
    ]

    checkpoint: dict = {}
    if not args.clean and CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"[RESUME] checkpoint con {len(checkpoint)} jueces")
        except (json.JSONDecodeError, IOError):
            print("[WARN] checkpoint corrupto, empezando de cero")

    print("\nAuditando con ambos jueces en paralelo...")
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(judges)) as pool:
        futs = {
            pool.submit(
                run_judge, units, model=j["model"], base_url=j["base_url"],
                key_rotator=j["key_rotator"], checkpoint=checkpoint,
                concurrency=args.concurrency,
            ): j["model"]
            for j in judges
        }
        for fut, model in futs.items():
            out[model] = fut.result()

    payload = {
        "units": {
            u["unit_id"]: {
                "mode": u["mode"], "seed": u["seed"], "window": u["window"],
                "tick": u["tick"], "position": u["position"], "n_prior": u["n_prior"],
            }
            for u in units
        },
        "judges": out,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    for model, res in out.items():
        ok = sum(1 for r in res.values() if r.get("ok"))
        print(f"  {model}: {ok}/{len(res)} auditorias validas")
    print(f"\nResultados en: {RESULTS_PATH}")
    print("Analiza con: python analyze_collapse.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
