# LangClaw: Homeostatic Regulation Prevents Context Collapse in Multi-Agent LLM Systems

## Project Summary

LangClaw is an experimental Multi-Agent System (MAS) framework designed to test the hypothesis that **endogenous homeostatic regulation (HRRL)** prevents context collapse more effectively than static orchestration in adversarial, zero-sum debate environments.

The core insight draws from biological homeostasis: rather than externally scheduling when agents speak (static orchestration), each agent has an internal *epistemic deficit* that grows over time and is reduced only by producing high-quality, structurally integrated arguments. A sigmoid activation function converts this deficit into a probability of participation, creating a self-regulating system where agents speak when they have something meaningful to contribute.

The framework includes a **comparative benchmark** (`benchmark.py`) that runs the same simulation under HRRL, Round-Robin, and Random orchestration modes with the same master seed, producing side-by-side comparison tables and charts.

---

## Theoretical Foundation

### Homeostatic Regulation via Reinforcement Learning (HRRL)

Each agent maintains an internal state variable, the **epistemic deficit** (delta), governed by three mechanisms:

**1. Decay (Drive Accumulation)**

The deficit increases linearly each tick, modelling the pressure to contribute:

$$\delta_{t+1} = \delta_t + \lambda$$

where lambda = 0.05 is the decay rate.

**2. Sigmoid Activation**

The probability of acting is computed via a sigmoid function:

$$p = \frac{1}{1 + e^{-k(\delta - \theta)}}$$

where k = 10 controls steepness and theta = 0.7 is the activation threshold. When delta < theta, agents are unlikely to act; when delta > theta, they are compelled to participate.

**3. Satiation (Drive Reduction)**

After producing an argument, the deficit is reduced proportionally to the argument's informational value:

$$\delta_{\text{new}} = \max(\varepsilon, \delta - \alpha \cdot \Delta\varphi^*)$$

where epsilon = 0.1 is the resting baseline, alpha = 0.5 is the learning rate, and Delta-Phi-star is the Integrated Information Theory proxy.

### IIT Phi-star Proxy

Computing true Integrated Information (Phi-star) is intractable. LangClaw uses a network-theoretic proxy:

$$\Phi^* \approx w_c \cdot C_{\text{betweenness}}(\text{target}) + w_{\text{cycle}} \cdot \mathbb{1}_{\text{cycle}} + w_{\text{div}} \cdot D_{\text{agents}}$$

with weights w_c = 0.35, w_cycle = 0.35, w_div = 0.30. This captures the intuition that an argument is informationally valuable when it:
- Targets a central node (betweenness centrality)
- Creates dialectical cycles (refutation loops)
- Bridges perspectives from different agents (diversity)

### In-Context Reinforcement Learning

Agents learn without fine-tuning. The last 3 experiences (state, action, reward) are injected into the LLM prompt, guiding the model to repeat argument styles that produced high Delta-Phi-star values.

---

## Architecture

```
langclaw_experiment/
├── langclaw/
│   ├── homeostasis.py    # EpistemicDrive: decay, sigmoid, satiation
│   ├── delp_graph.py     # ArgumentGraph: networkx DiGraph + Phi* proxy
│   ├── agent.py          # LangClawAgent: HRRL cycle + In-Context RL
│   ├── actions.py        # UtilitySelector + Tavily/fallback search
│   ├── memory.py         # Three-layer memory (episodic, semantic, working)
│   ├── budget.py         # APIBudget: per-agent rate limiting
│   ├── seeds.py          # Deterministic prime seed factory
│   ├── events.py         # Event dataclasses (tick, argument, shutdown)
│   ├── schemas.py        # Pydantic models: AgentAction, SimulationLog
│   └── simulation.py     # SotopiaEnvironment: 4-agent debate loop
├── main.py               # CLI entry point (single mode)
├── benchmark.py          # Comparative benchmark: HRRL vs baselines
├── dashboard.py          # Streamlit real-time visualization
├── paper_jaiio.tex       # JAIIO 2025/2026 submission (llncs format)
├── references.bib        # Bibliography (APA 7 format)
├── llncs.cls             # LNCS document class
├── requirements.txt
└── README.md
```

### Component Flow (HRRL mode)

```
For each tick (1..T):
  Broadcast TickElapsedEvent to all agents (async)
  Each agent concurrently:
    1. decay()                        → deficit increases
    2. get_activation_probability()   → sigmoid(deficit)
    3. random sample vs p             → activate or rest
    4. If activated:
       a. UtilitySelector picks action (DEBATE / SEARCH / READ / PASS)
       b. DEBATE: call LLM → parse JSON → add to graph → Phi* → satiate
       c. SEARCH: query Tavily (or fallback pool) → store in semantic memory
       d. READ: absorb recent working-memory entry as fact
    5. Broadcast NewArgumentEvents to peers
    6. Log metrics
```

### Orchestration Modes (for comparative benchmarks)

| Mode | Description |
|------|-------------|
| `hrrl` | Agents run as async coroutines, regulated by epistemic drive. No external scheduler. |
| `round-robin` | Every agent is forced to speak every tick (bypasses drive). Control baseline. |
| `random` | One randomly selected agent speaks each tick. Control baseline. |

---

## Experimental Setup

### Environment: Sotopia Zero-Sum Political Survival

Four agents in a political debate, split into two opposing factions. Each faction has one **Analytical** and one **Strategic** archetype, ensuring symmetric cognitive profiles across sides:

| Agent | Faction | Archetype | Objective |
|-------|---------|-----------|-----------|
| GOV-1 | Government | Analytical | Defend the government using data, statistics, and historical comparisons. Find factual inconsistencies in opposition arguments. |
| GOV-2 | Government | Strategic | Protect the official narrative by reframing criticism as out-of-context. Challenge premises, logic, and fallacies in opposition attacks. |
| OPP-1 | Opposition | Analytical | Attack the government using data, statistics, and historical comparisons. Find factual inconsistencies in government arguments. |
| OPP-2 | Opposition | Strategic | Dismantle the official narrative by reframing government defenses as insufficient. Challenge premises, logic, and fallacies in government justifications. |

This symmetric design ensures that differences in agent performance are attributable to the orchestration mode, not to role asymmetry.

### Search Capability

The SEARCH action uses the **Tavily Search API** for real web search when `TAVILY_API_KEY` is set in the `.env` file. When the key is absent, a static fallback knowledge pool is used, ensuring the simulation runs offline.

### Configuration

- **LLM**: OpenAI gpt-4o-mini
- **Iterations**: 50 ticks (30 for benchmarks)
- **Agents**: 4 (2v2 zero-sum, symmetric archetypes)
- **Decay rate (lambda)**: 0.05
- **Activation threshold (theta)**: 0.7
- **Sigmoid steepness (k)**: 10
- **Satiation rate (alpha)**: 2.0 (HRRL async path)
- **Baseline deficit (epsilon)**: 0.1
- **Initial deficit**: 0.5

### Metrics

- **tau-bench**: Fraction of debate turns that are logically consistent (argument connects to an existing node via a valid attack). tau = consistent_turns / total_debate_turns.
- **Avg Delta-Phi-star**: Mean informational value across all DEBATE actions.
- **Deficit trajectories**: Per-agent deficit over time.
- **Graph statistics**: Nodes, edges, density, components.
- **Per-agent debate count and quality**: Number of debates and average Delta-Phi-star per agent.

---

## Running the Benchmark

```bash
# Run all three modes with seed 42, 30 iterations each
python benchmark.py --model gpt-4o-mini --iterations 30 --seed 42

# Run only HRRL vs Round-Robin
python benchmark.py --modes hrrl round-robin --iterations 50 --seed 42

# Results are saved to benchmark_results/
#   benchmark_report.json     — aggregate metrics
#   logs_hrrl.json            — per-tick logs for HRRL
#   logs_round_robin.json     — per-tick logs for Round-Robin
#   logs_random.json          — per-tick logs for Random
#   tau_bench_comparison.html — interactive bar chart
#   debates_comparison.html   — debate volume & quality chart
#   deficit_evolution.html    — deficit trajectories per mode
#   agent_debates.html        — per-agent debate contribution
```

---

## Key Findings

1. **HRRL prevents context collapse**: High tau-bench scores indicate that the vast majority of debate turns produce logically connected arguments rather than isolated monologues.

2. **Self-regulating participation**: Without any external scheduler, agents naturally converge to a sustainable participation rhythm. The deficit-sigmoid loop creates an endogenous governor that prevents both over-participation (flooding) and under-participation (silence).

3. **Quality-driven satiation**: Agents producing higher-quality arguments (higher Delta-Phi-star) reduce their deficit faster, creating a natural selection pressure toward structurally integrated contributions.

4. **Symmetric role design eliminates bias**: The mirrored Analytical/Strategic archetypes across factions ensure that performance differences reflect orchestration dynamics, not role-inherent advantages.

5. **Comparative baseline**: The benchmark script enables direct comparison of tau-bench, debate volume, argument quality, and deficit dynamics across HRRL, Round-Robin, and Random modes.

---

## Limitations

1. **Phi-star is a proxy**: The IIT proxy uses betweenness centrality + cycle detection + agent diversity rather than true integrated information. This is explicitly acknowledged as a tractability constraint.

2. **Statistical power**: Rigorous conclusions require multiple runs with different seeds. The benchmark supports seeded reproducibility.

3. **LLM non-determinism**: Even with temperature=0.7, LLM outputs are stochastic. The HRRL mechanism itself uses random sampling for activation.

4. **Spanish-language debate**: The simulation was conducted in Spanish, which may affect LLM performance compared to English.

5. **Scale**: The 4-agent, 50-tick setup is small. Behavior at larger scales (more agents, longer horizons) is unknown.

---

## How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "OPEN_AI_API_KEY=sk-..." > .env
echo "TAVILY_API_KEY=tvly-..." >> .env   # optional, for real web search

# Run a single HRRL simulation
python main.py --model gpt-4o-mini --iterations 50 --output results.json

# Run the comparative benchmark
python benchmark.py --model gpt-4o-mini --iterations 30 --seed 42

# Launch the visualization dashboard
python -m streamlit run dashboard.py
```

---

## Citation

If referencing this experiment:

> Gerpe, P. (2026). *LangClaw: Homeostatic Regulation Prevents Context Collapse in Multi-Agent LLM Debate Systems*. Experimental framework and simulation results.
