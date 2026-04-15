# LangClaw — Homeostatic Multi-Agent Debate Framework

LangClaw is an experimental multi-agent system (MAS) to evaluate whether
**Homeostatic Regulation via Reinforcement Learning (HRRL)** sustains
context resilience better than exogenous orchestration (LangGraph router)
under equal heartbeat budgets.

## Current Experiment Scope

- 10 agents in a VSM structure (S1..S5 per faction).
- Deliberative cognitive loop per agent (`THINK -> PLAN -> EXECUTE -> OBSERVE`).
- Directed inter-agent messaging with FIPA-like communicative acts
  (`request`, `inform`, `propose`, `confirm`, `query`).
- Comparative modes: `hrrl` vs `langgraph`.
- Temporal metrics and AAF metrics, including defeat cycles as empirical outcomes.

## Repository Layout

```text
langclaw/
├── homeostasis.py      # Epistemic drive dynamics and activation gate
├── q_learner.py        # Online linear TD(0) Q-learning
├── delp_graph.py       # Argument graph + AAF + delta-phi proxy
├── agent.py            # Agent loop, prompts, actions, messaging
├── simulation.py       # Environment and orchestration modes
├── budget.py           # API hard/soft budget limits
├── actions.py          # Action utilities and search fallback tiers
├── langgraph_flow.py   # Exogenous LangGraph cognitive flow
└── schemas.py          # Pydantic logging/action schemas
```

Top-level scripts:

- `calibrate_hyperparams.py`: micro-simulation calibration with checkpoint/resume.
- `benchmark.py`: multi-seed benchmark with checkpoint/resume.
- `run_full_experiment.py`: detached supervisor (calibration -> benchmark) with
  status/event/state logs.

## Local Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure environment

Create `.env`:

```bash
OPEN_AI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...   # optional
```

### 3) Run the full experiment (detached, resilient)

```bash
python run_full_experiment.py --detach \
  --project-root . \
  --calibration-ticks 10 \
  --calibration-seed 42 \
  --calibration-api-hard-limit 200 \
  --iterations 80 \
  --seeds 7 17 42 123 256 \
  --benchmark-api-hard-limit 500 \
  --benchmark-output-dir benchmark_results_v7 \
  --status-file experiment_status.json \
  --log-file experiment_run.log \
  --events-file experiment_events.jsonl \
  --state-file experiment_state.txt \
  --log-level WARNING
```

### 4) Monitor run state

```bash
python run_full_experiment.py --status
```

```bash
# Windows PowerShell
Get-Content .\experiment_state.txt
Get-Content .\experiment_events.jsonl -Tail 20
Get-Content .\experiment_run.log -Tail 60
```

## Docker (Reproducible Setup)

### Build image

```bash
docker build -t langclaw:latest .
```

### Run full experiment with Compose

```bash
docker compose up --build
```

Notes:

- `docker-compose.yml` mounts the project directory into `/app`, so outputs,
  checkpoints, and logs persist on host.
- Set `OPEN_AI_API_KEY` in local `.env` before running compose.
- Compose command runs foreground inside container; use Docker restart policies
  externally if you want daemon-level auto-restart semantics.

## Recovery Behavior

- `calibrate_hyperparams.py` stores checkpoint progress per micro-run.
- `benchmark.py` stores checkpoint progress per `(mode, seed)` pair.
- `run_full_experiment.py` marks explicit states:
  - `RUNNING`
  - `PAUSED_RATE_LIMIT`
  - `FAILED`
  - `COMPLETED`

If API quota/rate limit is hit, run pauses with saved progress; re-run the same
command after restoring quota to resume.

## Dashboard

```bash
python -m streamlit run dashboard.py
```

