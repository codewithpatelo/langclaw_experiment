# Driveplexity — Experiment Summary

> Concise summary of the artefact accompanying the JAIIO 2026 paper
> *"Driveplexity: Endogenous Activation via Homeostasis in LLM-based
> Multi-Agent Debate."*
>
> For installation, reproduction, Docker setup, output formats,
> recovery, troubleshooting and AI tooling disclosure, see
> [`README.md`](README.md).

---

## 1. Research line

Can an LLM agent with *internal regulatory state* sustain valuable,
coherent action without an external planner deciding when it
participates? In adversarial, zero-sum debates with asymmetric
information, the claim under examination is that **endogenous
homeostatic activation** (Driveplexity) is more robust against
**context collapse** — the loss of dialectical coherence over
extended interaction — than **exogenous orchestration** (a
deterministic LangGraph router), when both are evaluated under
**matched temporal budgets** (same heartbeats, same per-agent
capabilities, same memory infrastructure).

The contrast is not "Driveplexity produces more debates" (it does,
by design) but "for the same heartbeat budget, Driveplexity sustains
coherence per debate while LangGraph degrades — and the effect
survives a volume-matched comparison."

---

## 2. Theoretical core (AAH)

The **Axioms of Homeostatic Autonomy** (`AAH`, previously `T_HA`)
state the minimal ontological commitments of the design. Full
statements and formalisms live in [`guide/TAH_axiomas.md`](guide/TAH_axiomas.md)
and in the paper, §*Axiomas de Autonomía Homeostática*.

- **A1 · Autonomy** (Ashby, 1956; Beer, 1972). The agent's policy
  depends on its internal state: `π_i = π_i(δ_i(t))`.
- **A2 · Drive** (Sterling & Eyer, 1988). An internal drive grows
  with deviation from equilibrium and monotonically determines
  activation probability: `D(ε)=0`, `D'(δ_i) > 0 ∀ δ_i > ε`,
  `p_i(t) = σ(D(δ_i(t)))`.
- **A3 · Quality gate** (Ryan & Deci, 2001). An action is valuable
  if it produces an effective change in the regulated environment;
  the drive decreases proportionally:
  `Δδ_i = −α · g(e(t), e(t+1))`, `α > 0`.

A derived prediction (**Pr1**) follows: an agent whose drive grows
with perturbation and only decreases after effective change is
expected to display sustained quality action, with activation
probability increasing in the deficit. The experimental system
**Driveplexity** operationalises A1–A3 in an MAS-LLM setting; the
mechanistic equations implemented in code are:

- **Decay (drive accumulation):** `δ_{t+1} = δ_t + λ`, with `λ = 0.05`.
- **Sigmoid activation gate:** `p = 1 / (1 + exp(−k(δ − θ)))`, with
  `k = 10`, `θ = 0.7`.
- **Satiation (drive reduction):** `δ_new = max(ε, δ − α · Δφ*)`, with
  `ε = 0.1` and `α` taken from `calibration_results.json`.
- **Q-learner (experimental extension):** linear TD(0) over 4
  normalised features, with TD-error and weight clipping plus L2
  regularisation; warm-starts the action policy. Its convergence
  under the deadly triad is reported as an empirical open question.

The `Δφ*` surrogate is a network-theoretic proxy combining target
betweenness centrality, a dialectical-cycle indicator and
inter-agent diversity (true Integrated Information is intractable;
this is explicitly flagged as a limitation).

---

## 3. Setup at a glance

| dimension                 | value                                                          |
|---------------------------|----------------------------------------------------------------|
| Agents                    | 10 (2 factions × 5 VSM subsystems S1–S5)                       |
| Cognitive loop            | event-driven `THINK → PLAN → EXECUTE → OBSERVE` per heartbeat  |
| Inter-agent messaging     | directed FIPA-like (`request`, `inform`, `propose`, `confirm`, `query`) |
| Actions                   | `DEBATE`, `SEARCH`, `READ`, `MESSAGE`, `PASS`                   |
| Memory                    | 3-layer (episodic / semantic / working), shared across modes   |
| LLM backbone              | OpenAI `gpt-5-nano-2025-08-07`                                 |
| Heartbeats per run        | 80 (canonical paper config)                                    |
| Main conditions           | `Driveplexity (hrrl)` vs `LangGraph`                           |
| Controls                  | `Driveplexity-no-Q` (ablation), `LangGraph-informed` (fair baseline) |
| Evaluation seeds          | `{7, 17, 99, 123, 256}` (`n = 5` paired)                       |
| Calibration-only seed     | `{42}` — excluded from evaluation                              |

Paper-relevant performance measures, defined in
`langclaw/metrics.py`, `langclaw/core_metric.py` and
`tools/volume_matched_analysis.py`:

- AAF acceptance ratio and its temporal slope (structural
  independence from the drive sensor).
- Top-speaker share (monopolisation proxy).
- Peer-Reference Rate (PRR\_G) and Initiative Ratio (IR).
- CORE (Conversational Robustness Evaluation).
- Δφ\* mean, slope, and per-agent dispersion (auxiliary; partial
  circularity with the sensor).
- Deliberative density (debates per heartbeat).
- Mean reward (drive reduction).
- Volume-matched variants of the above, on first-K and last-K
  windows.

---

## 4. Preliminary results (paper submission)

Magnitudes below reflect the paper's Table 1 (short version). Mean
± sample standard deviation.

| Metric                        | DPLXY            | LangGraph         | DPLXY-no-Q (n=1) | LG-informed (n=2)         |
|-------------------------------|------------------|-------------------|------------------|---------------------------|
| Total debates                 | `382 ± 6`        | `71 ± 6`          | `604`            | `75 ± 0`                  |
| Density (deb./hb.)            | `4.78`           | `0.89`            | `7.55`           | `0.94`                    |
| Top share (vm-1)              | `0.131 ± 0.008`  | `0.342 ± 0.046`   | `0.106` †        | `0.260 ± 0.010` †          |
| Top share (vm-ℓ)              | `0.154 ± 0.012`  | `0.342 ± 0.046`   | —                | —                          |
| AAF acc. (vm-1)               | `0.646 ± 0.043`  | `0.565 ± 0.022`   | `0.729` †        | `0.540 ± 0.010` †          |
| AAF acc. (vm-ℓ)               | `0.708 ± 0.061`  | `0.565 ± 0.022`   | —                | —                          |

† ablation metrics are reported on the *full* (untruncated) run.

Interpretation, hedged:

- Direction holds across all five seeds; Wilcoxon signed-rank
  `W = 0`, `p = 0.0625` is the **mechanical floor** for `n = 5`
  and does not cross `α_adj = 0.025` independently of effect size.
  Results are reported as descriptive / directional, not
  confirmatory.
- Top-speaker share drops from `~1/3` (LangGraph) to `~1/8`
  (Driveplexity), with dispersion one order of magnitude smaller
  (`σ 0.006` vs `0.046`) — compatible with a regime where each
  agent contributes when its `δ_i` justifies it.
- Volume-matched AAF acceptance in both windows is consistent with
  Driveplexity sustaining coherence rather than merely trading
  coherence for volume.
- Ablations (anecdotal, `n = 1, 2`): `DPLXY-no-Q` reproduces or
  intensifies the macro pattern; `LangGraph-informed` does not
  close the gap. The parsimonious reading is compatible with the
  effect residing in the **regulatory closure** (A2+A3) rather
  than in learning or information access.

---

## 5. Reproducibility

- Pinned `requirements.txt` and a `python:3.11-slim` Dockerfile.
- Deterministic seed factory (`langclaw/seeds.py`).
- Per-tick checkpoints in calibration and benchmark.
- Detached supervisor with watchdog auto-restart
  (`run_full_experiment.py`, `final_runner.py`).
- Per-run health reports with optional LLM explanation of red
  flags.
- Volume-matched analysis script (`tools/volume_matched_analysis.py`)
  with per-seed CSV and aggregate JSON outputs.

The minimal reproduction commands are documented in
[`README.md` §5](README.md#5-reproducing-the-paper-end-to-end).

---

## 6. Acknowledged limitations

1. `Δφ*` is a tractability surrogate, not Integrated Information
   itself.
2. Calibration was performed on a single seed (`42`) due to
   compute constraints; leakage is mitigated by excluding `42`
   from evaluation. `k`-fold calibration is future work.
3. `n = 5` paired seeds at submission time; statistical claims
   are preliminary and reported as descriptive / directional.
   Ablations are anecdotal (`n = 1, 2`) and their role is to
   bound alternative readings, not to establish effects.
4. The TD(0) loop does not converge (mean reward negative) — the
   paper acknowledges this explicitly and scopes the Q-learner as
   an experimental extension, not as a claim.
5. LLM stochasticity is bounded by temperature but not eliminated.
6. The simulation runs in Spanish, mirroring the paper's evaluation
   context; behaviour in other languages is unstudied.
7. Generalisation beyond `gpt-5-nano` and beyond zero-sum
   adversarial deliberation remains an open question.

A planned post-hoc LLM-as-judge protocol (two judges with an
AHP-weighted rubric, blind anonymisation, Krippendorff's α, Linear
Mixed-Effects Model with Bonferroni correction) is described in the
paper's *Acciones futuras* section and is implemented incrementally
in `tools/`.
