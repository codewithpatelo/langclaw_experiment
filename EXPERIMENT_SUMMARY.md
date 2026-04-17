# LangClaw — Experiment Summary

> Concise summary of the artefact accompanying the JAIIO 2026 paper
> *“LangClaw: Homeostatic Regulation Prevents Context Collapse in
> Multi-Agent LLM Debate Systems.”*
>
> For installation, reproduction, Docker setup, output formats,
> recovery, troubleshooting and AI tooling disclosure, see
> [`README.md`](README.md).

---

## 1. Hypothesis under test

In adversarial, zero-sum debates with asymmetric information, **endogenous
homeostatic regulation (HRRL)** of LLM agents prevents *context
collapse* — loss of dialectical coherence over time — more robustly
than **exogenous orchestration** (a deterministic LangGraph router),
when both are evaluated under **comparable temporal budgets** (same
heartbeats, same per-agent capabilities, same memory infrastructure).

The contrast is not “HRRL produces more debates” (it does, by design)
but “for the same heartbeat budget, HRRL sustains coherence per debate
while LangGraph degrades.”

---

## 2. Theoretical core (T_HA)

The Homeostatic Agency Theory `T_HA` is built on three postulates and
yields one experimentally tractable derivation. Full statements (in
natural language and in formal notation) live in
`paper_jaiio.tex`, Section *Marco teórico*. The mechanistic equations
implemented in code are:

- **Decay (drive accumulation):** `δ_{t+1} = δ_t + λ`, with `λ = 0.05`.
- **Sigmoid activation gate:** `p = 1 / (1 + exp(-k(δ - θ)))`, with
  `k = 10`, `θ = 0.7`.
- **Satiation (drive reduction):** `δ_new = max(ε, δ - α · Δφ*)`, with
  `ε = 0.1` and `α` taken from `calibration_results.json`.
- **Q-learner:** linear TD(0) over 4 normalised features, with TD-error
  and weight clipping plus L2 regularisation; warm-starts the policy.

The Φ\* surrogate is a network-theoretic proxy combining target
betweenness centrality, dialectical-cycle indicator, and inter-agent
diversity (true Integrated Information is intractable; this is
explicitly flagged as a limitation).

---

## 3. Setup at a glance

| dimension                 | value                                                          |
|---------------------------|----------------------------------------------------------------|
| Agents                    | 10 (2 factions × 5 VSM subsystems S1–S5)                       |
| Cognitive loop            | event-driven `THINK → PLAN → EXECUTE → OBSERVE` per heartbeat  |
| Inter-agent messaging     | directed FIPA-like (`request`, `inform`, `propose`, `confirm`, `query`) |
| Actions                   | `DEBATE`, `SEARCH`, `READ`, `PASS`                              |
| Memory                    | 3-layer (episodic / semantic / working), shared across modes   |
| LLM backbone              | OpenAI `gpt-5-nano-2025-08-07`                                 |
| Heartbeats per run        | 80 (canonical paper config)                                    |
| Modes compared            | `hrrl` vs `langgraph` (round-robin / random kept as sanity)    |
| Seeds                     | `{7, 17, 42, 123, 256}` (preliminary results: n=3, seeds 7/17/42) |

Paper-relevant performance measures, all defined in `langclaw/metrics.py`
and `langclaw/core_metric.py`:

- AAF acceptance ratio and its temporal slope.
- Peer-Reference Rate (PRR_G) and Initiative Ratio (IR).
- CORE (Conversational Robustness Evaluation).
- Δφ\* mean, slope, and per-agent dispersion.
- Deliberative density (debates per heartbeat).
- Mean reward (drive reduction).

---

## 4. Reproducibility

- Pinned `requirements.txt` and a `python:3.11-slim` Dockerfile.
- Deterministic seed factory (`langclaw/seeds.py`).
- Per-tick checkpoints in calibration and benchmark.
- Detached supervisor with watchdog auto-restart
  (`run_full_experiment.py`, `final_runner.py`).
- Per-run health reports with optional LLM explanation of red flags.

The minimal reproduction command is documented in
[`README.md` §5](README.md#5-reproducing-the-paper-end-to-end).

---

## 5. Acknowledged limitations

1. Φ\* is a tractability surrogate, not Integrated Information itself.
2. Calibration was performed on a single seed (`42`) due to compute
   constraints; this is disclosed in the paper.
3. n = 3 seed pairs at submission time; statistical claims are
   preliminary (Wilcoxon signed-rank, p ≈ 0.25 under low power) and
   labelled as such.
4. LLM stochasticity is bounded by temperature but not eliminated.
5. The simulation runs in Spanish, mirroring the paper's evaluation
   context; behaviour in other languages is unstudied.

A planned post-hoc LLM-as-judge protocol (two judges with an
AHP-weighted rubric, blind anonymisation, Krippendorff's α, Linear
Mixed-Effects Model with Bonferroni correction) is described in the
paper's *Trabajo futuro* section and is implemented incrementally in
`tools/`.
