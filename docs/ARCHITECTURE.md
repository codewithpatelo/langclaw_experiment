# Driveplexity — Architecture

This document is a technical tour of the codebase for someone who
wants to read, modify or extend the experiment. For the scientific
framing see [`../paper_jaiio_short.tex`](../paper_jaiio_short.tex)
and [`../EXPERIMENT_SUMMARY.md`](../EXPERIMENT_SUMMARY.md). For
installation and reproduction see [`../README.md`](../README.md).

---

## 1. Package layout (`langclaw/`)

The Python package is named `langclaw/` for historical reasons (the
project's internal codename during development). It is the concrete
implementation of **Driveplexity** described in the paper.

```text
langclaw/
├── homeostasis.py       # HRRL-style drive, sigmoid gate, satiation
├── q_learner.py         # Linear TD(0) Q-learner (experimental extension)
├── delp_graph.py        # Argument graph (AAF) and Δφ* proxy
├── agent.py             # Per-agent cognitive loop, prompts, FIPA messaging
├── simulation.py        # Environment, event loop, both orchestration modes
├── langgraph_flow.py    # LangGraph baseline router
├── router.py            # Inter-agent message routing
├── router_informed.py   # Fair-baseline router that receives DPLXY's features
├── memory.py            # Three-layer memory (episodic / semantic / working)
├── budget.py            # Hard and soft API rate limits
├── actions.py           # Action utilities, StimulusEvaluator, search fallback
├── core_metric.py       # CORE temporal coherence metric
├── metrics.py           # PRR_G, IR, AAF acceptance, slopes
├── events.py            # Tick / argument / shutdown events
├── schemas.py           # Pydantic logging schemas
└── seeds.py             # Deterministic prime-seed factory
```

---

## 2. Cognitive loop

Each agent runs an event-driven `THINK → PLAN → EXECUTE → OBSERVE`
cycle per heartbeat, *only when triaged*. The triage stage is where
the two orchestration strategies diverge:

- **Driveplexity (endogenous).** `homeostasis.py` maintains a
  per-agent deficit `δ_i`. Each heartbeat it decays upward by `λ`;
  if the `StimulusEvaluator` in `actions.py` detects relevant
  perturbations it adds weighted contributions. A sigmoid
  `p = σ(k(δ − θ))` turns `δ` into an activation probability,
  sampled via Bernoulli; the event that advances the loop is
  therefore *internal state crossing threshold*, not a scheduler
  pulse.
- **LangGraph (exogenous).** `langgraph_flow.py` runs a deterministic
  neutral-moderator router that chooses the next speaker from the
  ten agents (with round-robin fallback). The triage event is the
  router's decision.

Once triaged, the rest of the loop is identical across modes: agents
`THINK` (produce an internal plan via the LLM), `PLAN` (select a
concrete action from `DEBATE / SEARCH / READ / MESSAGE / PASS`),
`EXECUTE` (run it and emit the required events), `OBSERVE` (update
memory and, in Driveplexity, the deficit and Q-learner).

---

## 3. Homeostatic regulation (`homeostasis.py`)

Implements A1–A3 from the paper in three coupled pieces:

- **Drive function** `D(δ) = (δ − ε)^m` (default `m = 2`). Satisfies
  `D(ε) = 0` and `D'(δ) > 0` for `δ > ε`, so the drive grows
  super-linearly with deviation from the set-point `ε`.
- **State transitions.** `δ` increases via basal decay (`δ += λ`)
  and stimulus contributions, and decreases via quality-proportional
  satiation `δ ← max(ε, δ − α · Δφ*)`.
- **Activation gate.** Bernoulli sample over `p = σ(k(δ − θ))` with
  `k = 10`, `θ = 0.7`.

Default hyperparameters are tuned so that: an agent starting at
`δ₀ = 0.5` reaches 50% activation around tick 4; a good debate
action (`Δφ* ≈ 0.15`) brings `δ` back to `≈ 0.4`, generating the
act-rest oscillation that characterises homeostasis.

---

## 4. Q-learning extension (`q_learner.py`)

A linear TD(0) Q-learner over four normalised state features,
intended to *warm-start* preferences between actions (not to replace
regulation). TD-error and weights are clipped, L2 regularisation is
applied, and off-policy linear function approximation places the
learner in a **deadly triad** regime. The paper reports
non-convergence (`mean reward ≈ −2.03`) as an expected empirical
observation and scopes the Q-learner as an **experimental extension**
rather than a claim.

The `Driveplexity-no-Q` ablation (`run_ablation.py --variant no_q`)
bypasses the Q-learner and preserves only the homeostatic closure;
the paper uses it to argue that the observed macro pattern survives
the learner's failure.

---

## 5. Argument graph and Δφ\* (`delp_graph.py`)

The dialectical state is modelled as an Abstract Argumentation
Framework (AAF; Dung, 1995). Each debate move adds nodes and attack
edges. Two quantities come out of it:

- **AAF acceptance ratio.** Grounded semantics applied to the
  current graph; reported as the main structural-coherence metric,
  chosen because it is structurally independent of the homeostatic
  sensor.
- **Δφ\* proxy.** Network-theoretic surrogate combining target
  betweenness centrality, a dialectical-cycle indicator and
  inter-agent diversity. Feeds the satiation update and serves as
  an auxiliary diagnostic. Partial circularity with the sensor is
  acknowledged; that's why the primary metrics in the paper table
  are structurally independent.

---

## 6. Stimulus evaluation (`actions.py`)

`StimulusEvaluator` is a deterministic component that fuses five
structural features — factional relevance, attacked-node
centrality, novelty, unanswered pressure, semantic memory recall —
into a single weighted score that modulates `δ_i` before the sigmoid
gate. The same features are exposed to `router_informed.py` as a
JSON payload so that the `LangGraph-informed` fair baseline can make
its routing decisions with the same information DPLXY uses
internally.

---

## 7. Orchestration modes (`simulation.py`)

`simulation.py` is the single entry point that runs the event loop
in any mode. It instantiates ten agents (two factions × five VSM
subsystems), wires them to the shared `memory.py`, `budget.py`,
`router.py` and logging infrastructure, and delegates triage to one
of:

- `homeostasis.py` (modes `hrrl` / `hrrl_no_q`).
- `langgraph_flow.py` (modes `langgraph` / `langgraph_informed`).
- `simulation.py`'s sanity schedulers (modes `round-robin`,
  `random`).

Cross-mode invariants (identical in all modes):

- Per-agent capabilities and tool access.
- Heartbeat budget.
- Memory architecture and retrieval.
- Action set and prompts.
- Logging and checkpointing.

What differs across modes is **strictly the triage** — which agent's
cognitive loop fires on each heartbeat.

---

## 8. Memory (`memory.py`)

Three-layer memory shared across modes:

- **Working memory.** Per-agent scratchpad for the current
  heartbeat.
- **Episodic memory.** Append-only log of events, scoped by run.
- **Semantic memory.** Summarised long-term facts, used by the
  StimulusEvaluator's recall feature and by the LLM when composing
  `DEBATE` / `MESSAGE` utterances.

---

## 9. Budget (`budget.py`)

Two soft and one hard limit per run, to keep experiments bounded and
observable:

- **Hard cap on API calls.** Run aborts cleanly once reached.
- **Soft per-heartbeat cap.** Warning when approached.
- **Rate-limit backoff.** OpenAI 429s pause the worker in state
  `PAUSED_RATE_LIMIT`; the supervisor (`run_full_experiment.py`)
  resumes automatically on next launch.

Importantly, **tokens are not equalised across modes** (doing so
would force DPLXY to suppress its own activations, collapsing the
very variable under study). The paper uses **matched heartbeats**
and reports `router_call_count` for transparency.

---

## 10. Metrics (`metrics.py`, `core_metric.py`, `tools/`)

The post-run analysis pipeline computes, for each `(mode, seed)`
pair:

- **Structural (primary).** AAF acceptance ratio, top-speaker share,
  debate density.
- **Reference-based.** `PRR_G`, `IR`.
- **Temporal.** CORE slope.
- **Homeostatic auxiliaries.** `Δφ*` mean / slope / dispersion,
  mean reward.

The volume-matched control lives in
`tools/volume_matched_analysis.py`: for each seed it truncates
DPLXY to `K = N_LG` debates and recomputes the structural metrics
on two windows (first-`K` and last-`K`). The paper's Table 1
includes the dual `vm-1` / `vm-ℓ` rows.

Per-agent statistics (top-speaker share, distribution) are computed
by `tools/agent_stats.py`.

---

## 11. Seeds and determinism (`seeds.py`)

Every random source — Python `random`, `numpy`, LLM sampling salt,
agent-id assignment — derives deterministically from a master seed
via the prime factory in `seeds.py`. Re-running the same command
with the same seed and the same `requirements.txt` reproduces the
same trajectory modulo OpenAI non-determinism (temperature `> 0`).
