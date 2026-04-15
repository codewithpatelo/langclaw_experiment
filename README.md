# LangClaw — Homeostatic Multi-Agent Debate Framework

Experimental Multi-Agent System (MAS) framework for studying whether
**Homeostatic Regulation via Reinforcement Learning (HRRL)** prevents
context collapse better than static orchestration in a zero-sum
Sotopia-style debate.

## Architecture

```
langclaw/
├── homeostasis.py   # EpistemicDrive — sigmoid activation, decay, satiation
├── delp_graph.py    # ArgumentGraph — networkx digraph + IIT Φ* proxy
├── agent.py         # LangClawAgent — HRRL cycle + In-Context RL
├── actions.py       # UtilitySelector + Tavily/fallback search
├── memory.py        # Three-layer memory (episodic, semantic, working)
├── budget.py        # APIBudget — per-agent rate limiting
├── seeds.py         # Deterministic prime seed factory
├── events.py        # Event dataclasses (tick, argument, shutdown)
├── schemas.py       # Pydantic models for structured LLM output
└── simulation.py    # SotopiaEnvironment — 4-agent zero-sum loop
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
# .env file
OPEN_AI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...   # optional — enables real web search
```

### 3. Run via CLI (OpenAI API)

```bash
python main.py --model gpt-4o-mini --iterations 50 --output results.json
```

### 4. Run via CLI (Ollama)

```bash
python main.py --base-url http://localhost:11434/v1 --model llama3 --api-key ollama
```

### 5. Run the Comparative Benchmark

```bash
python benchmark.py --model gpt-4o-mini --iterations 30 --seed 42
# Results saved to benchmark_results/
```

### 6. Launch the Dashboard

```bash
python -m streamlit run dashboard.py
```

## HRRL Overview

Each agent has an internal *epistemic deficit* that grows over time (decay).
A sigmoid function maps deficit to action probability:

$$p = \frac{1}{1 + e^{-k(\delta - \theta)}}$$

When an agent acts, the quality of its contribution (Δφ\*) reduces the deficit:

$$\delta_{\text{new}} = \max(\varepsilon,\; \delta - \alpha \cdot \Delta\varphi^*)$$

High-quality arguments (those that target central nodes, create dialectical cycles,
and bridge multiple agents) yield higher Δφ\*, satisfying the drive. Isolated
monologues yield Δφ\* ≈ 0, leaving the deficit unchanged.

## Agents (Zero-Sum Political Survival)

Symmetric cognitive archetypes (Analytical / Strategic) across both factions:

| ID    | Faction     | Archetype  | Objective                                              |
|-------|-------------|------------|--------------------------------------------------------|
| GOV-1 | Government  | Analytical | Defend with data, statistics, historical comparisons   |
| GOV-2 | Government  | Strategic  | Reframe criticism, challenge premises and fallacies    |
| OPP-1 | Opposition  | Analytical | Attack with data, statistics, historical comparisons   |
| OPP-2 | Opposition  | Strategic  | Reframe defenses, challenge premises and fallacies     |

## Orchestration Modes

| Mode         | Description                                  |
|--------------|----------------------------------------------|
| `hrrl`       | Endogenous regulation — agents decide when to speak |
| `round-robin`| Every agent speaks every tick (static baseline) |
| `random`     | One random agent per tick (stochastic baseline) |

## Metrics

- **τ-bench**: Fraction of debate turns that are logically consistent
  (argument connects to an existing node via a valid attack).
- **Avg Δφ\***: Mean informational value of DEBATE contributions.
- **Deficit trajectories**: Per-agent deficit over time.
- **Graph density**: Argument graph connectivity evolution.
- **Per-agent debate count**: Contribution distribution across agents.
