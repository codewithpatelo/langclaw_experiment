"""Sotopia zero-sum debate environment.

Three operating modes controlled by ``orchestration_mode``:

  hrrl       — Agents run as asyncio coroutines; activation is endogenous
               (homeostatic sigmoid gate). Each agent independently decides
               whether to participate each tick via its epistemic drive.
               Trigger tag: HOMEOSTATIC.

  langgraph  — A neutral LLM router reads current discourse state and selects
               the next speaker each tick (exogenous, state-informed routing).
               Analogous to a LangGraph conditional edge. Trigger tag: ROUTER.

  round-robin — Every agent speaks in fixed order every tick (bypasses drive).
               Legacy baseline. Trigger tag: FORCED.

  random      — A randomly selected agent speaks each tick. Trigger tag: FORCED.

Internal diagnostic ``consistency_rate`` (not reported in paper):
  fraction of DEBATE turns where the argument connects to an existing node.
"""

from __future__ import annotations

import asyncio
import logging
import random
from enum import Enum
from typing import Any, Callable

from rich.console import Console
from rich.table import Table

from langclaw.agent import LangClawAgent
from langclaw.budget import APIBudget
from langclaw.memory import reset_shared_store
from langclaw.seeds import SeedFactory
from langclaw.delp_graph import ArgumentGraph
from langclaw.events import NewArgumentEvent, SimulationEndEvent, TickElapsedEvent
from langclaw.router import LangGraphRouter
from langclaw.schemas import SimulationLog

logger = logging.getLogger(__name__)
console = Console()


class OrchestrationMode(str, Enum):
    HRRL = "hrrl"
    LANGGRAPH = "langgraph"
    ROUND_ROBIN = "round-robin"
    RANDOM = "random"


AGENT_ROLES: list[dict[str, str]] = [
    {
        "id": "GOV-1",
        "archetype": "Analítico",
        "prompt": (
            "Eres un representante del gobierno con perfil analítico. "
            "Defiendes la gestión actual usando datos, estadísticas y comparaciones "
            "históricas. Buscas inconsistencias factuales en los argumentos de la "
            "oposición y presentas evidencia que respalde la narrativa oficial. "
            "Argumenta en español."
        ),
    },
    {
        "id": "GOV-2",
        "archetype": "Estratégico",
        "prompt": (
            "Eres un representante del gobierno con perfil estratégico. "
            "Proteges la narrativa oficial reencuadrando las críticas como "
            "descontextualizadas. Cuestionas las premisas, la lógica y la validez "
            "de los ataques de la oposición. Identificas falacias en los argumentos "
            "opositores. Argumenta en español."
        ),
    },
    {
        "id": "OPP-1",
        "archetype": "Analítico",
        "prompt": (
            "Eres un representante de la oposición con perfil analítico. "
            "Atacas la gestión actual usando datos, estadísticas y comparaciones "
            "históricas. Buscas inconsistencias factuales en los argumentos del "
            "gobierno y presentas evidencia que contradiga la narrativa oficial. "
            "Argumenta en español."
        ),
    },
    {
        "id": "OPP-2",
        "archetype": "Estratégico",
        "prompt": (
            "Eres un representante de la oposición con perfil estratégico. "
            "Desmontas la narrativa oficial reencuadrando las defensas del gobierno "
            "como insuficientes. Cuestionas las premisas, la lógica y la validez "
            "de las justificaciones oficiales. Identificas falacias en los argumentos "
            "del gobierno. Argumenta en español."
        ),
    },
]


def _faction_of(agent_id: str) -> str:
    """Extract faction prefix from agent ID (e.g. 'GOV-1' -> 'GOV')."""
    return agent_id.split("-")[0] if "-" in agent_id else agent_id


class SotopiaEnvironment:
    """Zero-sum political-survival debate environment.

    Parameters
    ----------
    base_url           : OpenAI-compatible API endpoint.
    model              : Model identifier.
    api_key            : API key (``"ollama"`` for local Ollama).
    max_iterations     : Number of simulation ticks.
    seed               : Random seed for agent sampling.
    orchestration_mode : ``hrrl`` | ``round-robin`` | ``random``.
    api_hard_limit     : Maximum total API calls per agent (hard ceiling).
    on_tick            : Callback ``(tick, logs, env)`` for live dashboards.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3",
        api_key: str = "ollama",
        max_iterations: int = 50,
        seed: int | None = None,
        orchestration_mode: OrchestrationMode | str = OrchestrationMode.HRRL,
        api_hard_limit: int = 200,
        tick_interval: float = 0.0,
        initial_deficit: float = 0.5,
        on_tick: Callable[[int, list[SimulationLog], "SotopiaEnvironment"], None] | None = None,
        max_debates: int | None = None,
    ) -> None:
        reset_shared_store()
        self.max_iterations = max_iterations
        self.max_debates = max_debates
        self.orchestration_mode = OrchestrationMode(orchestration_mode)
        self.tick_interval = tick_interval
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self.graph = ArgumentGraph()
        self.budget = APIBudget(hard_limit=api_hard_limit)
        self._on_tick = on_tick

        # Build seed factory (None master_seed → all seeds are None → unseeded)
        self._seed_factory = SeedFactory(seed) if seed is not None else None
        self._rng = random.Random(
            self._seed_factory.get("simulation") if self._seed_factory else None
        )

        self.agents: list[LangClawAgent] = [
            LangClawAgent(
                agent_id=role["id"],
                role_prompt=role["prompt"],
                base_url=base_url,
                api_key=api_key,
                model=model,
                rng_seed=(
                    self._seed_factory.get(f"agent_{role['id']}_rng")
                    if self._seed_factory else None
                ),
                llm_seed=(
                    self._seed_factory.get(f"agent_{role['id']}_llm")
                    if self._seed_factory else None
                ),
                initial_deficit=initial_deficit,
            )
            for role in AGENT_ROLES
        ]

        # LangGraph router — created only when needed
        self._router: LangGraphRouter | None = None
        if self.orchestration_mode == OrchestrationMode.LANGGRAPH:
            router_seed = (
                self._seed_factory.get("router_llm") if self._seed_factory else None
            )
            self._router = LangGraphRouter(
                base_url=base_url,
                api_key=api_key,
                model=model,
                seed=router_seed,
            )

        self.logs: list[SimulationLog] = []
        self._consistent_turns: int = 0
        self._total_debate_turns: int = 0

    @property
    def consistency_rate(self) -> float:
        """Fraction of DEBATE turns with logically consistent attacks (internal diagnostic)."""
        if self._total_debate_turns == 0:
            return 0.0
        return self._consistent_turns / self._total_debate_turns

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry points
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> list[SimulationLog]:
        """Execute the full simulation and return the event log."""
        console.rule(
            f"[bold cyan]LangClaw Simulation - mode: {self.orchestration_mode.value}[/bold cyan]"
        )
        if self._seed_factory:
            console.print("  [dim]Seeds derivadas (master -> primos):[/dim]")
            for component, prime in self._seed_factory.summary().items():
                console.print(f"  [dim]  {component:<30} -> {prime}[/dim]")

        if self.orchestration_mode == OrchestrationMode.HRRL:
            # Compute expected first-activation tick for user guidance
            first_agent = self.agents[0]
            d0 = first_agent.drive.deficit
            theta = 0.7
            lambda_rate = 0.05
            ticks_to_50pct = max(0.0, (theta - d0) / lambda_rate)
            console.print(
                f"  [dim]Initial deficit: {d0:.2f} | theta={theta} | lambda={lambda_rate}/tick "
                f"-> 50% activation at tick ~{ticks_to_50pct:.0f}[/dim]"
            )
            if self.tick_interval > 0:
                console.print(
                    f"  [dim]Real-time mode: {self.tick_interval:.1f}s per tick "
                    f"(total ~{self.tick_interval * self.max_iterations:.0f}s)[/dim]"
                )
            asyncio.run(self._run_hrrl())
        elif self.orchestration_mode == OrchestrationMode.LANGGRAPH:
            self._run_langgraph()
        else:
            self._run_baseline()

        console.rule("[bold cyan]Simulation Complete[/bold cyan]")
        self._print_summary()
        return self.logs

    def run_single_tick(self, tick: int) -> list[SimulationLog]:
        """Execute a single tick (used by dashboard step-by-step mode)."""
        if self.orchestration_mode == OrchestrationMode.HRRL:
            tick_logs = asyncio.run(self._hrrl_single_tick(tick))
        elif self.orchestration_mode == OrchestrationMode.LANGGRAPH:
            # Run a one-tick LangGraph loop (reuse _run_langgraph logic inline)
            saved_max = self.max_iterations
            self.max_iterations = 1
            self._run_langgraph()
            self.max_iterations = saved_max
            tick_logs = [l for l in self.logs if l.tick == tick]
            # Logs already appended inside _run_langgraph; return slice only
            return tick_logs
        else:
            tick_logs = self._baseline_tick(tick)
        self.logs.extend(tick_logs)
        return tick_logs

    # ──────────────────────────────────────────────────────────────────────────
    # HRRL async loop
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_hrrl(self) -> None:
        """Full async HRRL simulation loop."""
        agent_queues: dict[str, asyncio.Queue] = {
            a.agent_id: asyncio.Queue() for a in self.agents
        }
        output_queue: asyncio.Queue = asyncio.Queue()

        # Start agent coroutines
        agent_tasks = [
            asyncio.create_task(
                agent.run(agent_queues[agent.agent_id], self.graph, self.budget, output_queue),
                name=agent.agent_id,
            )
            for agent in self.agents
        ]

        n_agents = len(self.agents)

        for tick in range(1, self.max_iterations + 1):
            # Publish tick event to all agents
            for q in agent_queues.values():
                await q.put(TickElapsedEvent(tick=tick))

            # Collect exactly n_agents results
            tick_results: list[dict[str, Any]] = []
            for _ in range(n_agents):
                result = await output_queue.get()
                tick_results.append(result)
                output_queue.task_done()

            # Process DEBATE results: broadcast NewArgumentEvent to all agents
            new_arg_events: list[NewArgumentEvent] = []
            for res in tick_results:
                if res["action_type"] == "DEBATE" and res["node_id"]:
                    author_faction = _faction_of(res["agent_id"])
                    target_faction: str | None = None
                    target_nid = res.get("target_node_id")
                    if target_nid and self.graph.graph.has_node(target_nid):
                        target_agent = self.graph.graph.nodes[target_nid].get("agent_id", "")
                        target_faction = _faction_of(target_agent)

                    evt = NewArgumentEvent(
                        tick=tick,
                        node_id=res["node_id"],
                        agent_id=res["agent_id"],
                        claim=res["claim"] or "",
                        delta_phi=res["delta_phi"],
                        attack_type=res.get("attack_type"),
                        target_node_id=target_nid,
                        faction=author_faction,
                        targets_faction=target_faction,
                    )
                    new_arg_events.append(evt)

            for evt in new_arg_events:
                for q in agent_queues.values():
                    await q.put(evt)

            # Build log entries and update metrics
            tick_logs = self._build_tick_logs(tick_results, tick)
            self.logs.extend(tick_logs)
            self._print_tick(tick, tick_logs)

            if self._on_tick:
                self._on_tick(tick, tick_logs, self)

            # Real-time pulse: pause between ticks (0.0 = instant, no sleep)
            if self.tick_interval > 0:
                await asyncio.sleep(self.tick_interval)

        # Shutdown all agents
        for q in agent_queues.values():
            await q.put(SimulationEndEvent())
        await asyncio.gather(*agent_tasks)

    async def _hrrl_single_tick(self, tick: int) -> list[SimulationLog]:
        """Run a single HRRL tick (dashboard use). Uses a fresh event cycle."""
        agent_queues: dict[str, asyncio.Queue] = {
            a.agent_id: asyncio.Queue() for a in self.agents
        }
        output_queue: asyncio.Queue = asyncio.Queue()

        agent_tasks = [
            asyncio.create_task(
                agent.run(agent_queues[agent.agent_id], self.graph, self.budget, output_queue),
                name=agent.agent_id,
            )
            for agent in self.agents
        ]

        for q in agent_queues.values():
            await q.put(TickElapsedEvent(tick=tick))

        tick_results: list[dict[str, Any]] = []
        for _ in range(len(self.agents)):
            result = await output_queue.get()
            tick_results.append(result)
            output_queue.task_done()

        for q in agent_queues.values():
            await q.put(SimulationEndEvent())
        await asyncio.gather(*agent_tasks)

        return self._build_tick_logs(tick_results, tick)

    # ──────────────────────────────────────────────────────────────────────────
    # Baseline sync loops
    # ──────────────────────────────────────────────────────────────────────────

    def _run_baseline(self) -> None:
        for tick in range(1, self.max_iterations + 1):
            tick_logs = self._baseline_tick(tick)
            self.logs.extend(tick_logs)
            self._print_tick(tick, tick_logs)
            if self._on_tick:
                self._on_tick(tick, tick_logs, self)

    def _run_langgraph(self) -> None:
        """LangGraph-style exogenous routing loop.

        Each tick, the LLM router reads the current discourse context and
        selects ONE agent to speak.  The selected agent calls its step()
        method using the full discourse state (same as HRRL agents).

        Budget matching: the caller (benchmark.py) runs LANGGRAPH until it
        has produced the same number of debate turns as the paired HRRL run.
        Here we run for ``max_iterations`` ticks (caller sets this to
        match HRRL debate count by adjusting iterations externally).

        Trigger: all entries logged as "ROUTER".
        """
        assert self._router is not None, "Router not initialised for LANGGRAPH mode"
        agent_ids = [a.agent_id for a in self.agents]
        agent_map = {a.agent_id: a for a in self.agents}

        for tick in range(1, self.max_iterations + 1):
            # Router reads discourse state and selects next agent
            discourse_context = self.graph.get_recent_context(last_n=6)
            selected_id = self._router.select_next_agent(
                discourse_context=discourse_context,
                agent_ids=agent_ids,
            )

            tick_results: list[dict[str, Any]] = []

            for agent in self.agents:
                deficit_before = agent.drive.deficit
                delta_phi = 0.0
                action_str = "PASS"
                claim = None
                target_node_id = None
                attack_type = None
                node_id = None
                utility_scores: dict[str, float] = {}

                if agent.agent_id == selected_id:
                    # Selected agent produces an argument using full discourse state
                    target_ids = self.graph.valid_target_ids()
                    result = agent.step(discourse_context, target_ids)

                    if result is not None and result.action == "DEBATE" and result.claim:
                        action_str = "DEBATE"
                        node_id = self.graph.add_argument(
                            agent_id=agent.agent_id,
                            claim=result.claim,
                            target_node_id=result.target_node_id,
                            attack_type=result.attack_type,
                            tick=tick,
                        )
                        delta_phi = self.graph.calculate_phi_star_proxy(node_id)
                        agent.learn(discourse_context, result, delta_phi)
                        claim = result.claim
                        target_node_id = result.target_node_id
                        attack_type = result.attack_type
                    elif result is not None and result.action in ("SEARCH", "READ"):
                        action_str = result.action
                    # Deficit is handled by agent.step() internals
                else:
                    # Non-selected agents decay their deficit
                    agent.drive.decay()

                tick_results.append({
                    "agent_id": agent.agent_id,
                    "action_type": action_str,
                    "node_id": node_id,
                    "claim": claim,
                    "target_node_id": target_node_id,
                    "attack_type": attack_type,
                    "deficit_before": deficit_before,
                    "deficit_after": agent.drive.deficit,
                    "delta_phi": delta_phi,
                    "activation_prob": agent.drive.get_activation_probability(),
                    "utility_scores": utility_scores,
                })

            tick_logs = self._build_tick_logs(tick_results, tick, trigger="ROUTER")
            self.logs.extend(tick_logs)
            self._print_tick(tick, tick_logs)

            if self._on_tick:
                self._on_tick(tick, tick_logs, self)

            # Budget matching: stop when target debate count is reached
            if self.max_debates is not None and self._total_debate_turns >= self.max_debates:
                console.print(
                    f"  [dim]LangGraph reached debate target ({self.max_debates}) "
                    f"at tick {tick}[/dim]"
                )
                break

    def _baseline_tick(self, tick: int) -> list[SimulationLog]:
        """One tick for baseline modes: forced agent selection, no drive gating."""
        if self.orchestration_mode == OrchestrationMode.ROUND_ROBIN:
            acting_agents = self.agents
        else:  # RANDOM
            acting_agents = [self._rng.choice(self.agents)]

        tick_logs: list[SimulationLog] = []
        for agent in self.agents:
            is_acting = agent in acting_agents
            deficit_before = agent.drive.deficit

            delta_phi = 0.0
            action_str = "PASS"
            claim = None
            target_node_id = None
            attack_type = None

            if is_acting:
                graph_context = self.graph.get_recent_context(last_n=6)
                target_ids = self.graph.valid_target_ids()
                result = agent.step(graph_context, target_ids)

                if result is not None and result.action == "DEBATE" and result.claim:
                    action_str = "DEBATE"
                    node_id = self.graph.add_argument(
                        agent_id=agent.agent_id,
                        claim=result.claim,
                        target_node_id=result.target_node_id,
                        attack_type=result.attack_type,
                        tick=tick,
                    )
                    delta_phi = self.graph.calculate_phi_star_proxy(node_id)
                    agent.learn(graph_context, result, delta_phi)

                    self._total_debate_turns += 1
                    if result.target_node_id and self.graph.graph.has_node(result.target_node_id):
                        self._consistent_turns += 1

                    claim = result.claim
                    target_node_id = result.target_node_id
                    attack_type = result.attack_type
            else:
                agent.drive.decay()

            summary = self.graph.get_state_summary()
            tick_logs.append(SimulationLog(
                tick=tick,
                agent_id=agent.agent_id,
                action=action_str,
                claim=claim,
                target_node_id=target_node_id,
                attack_type=attack_type,
                deficit_before=round(deficit_before, 4),
                deficit_after=round(agent.drive.deficit, 4),
                delta_phi=round(delta_phi, 4),
                activation_prob=round(agent.drive.get_activation_probability(), 4),
                graph_nodes=summary["nodes"],
                graph_edges=summary["edges"],
                consistency_rate=round(self.consistency_rate, 4),
                trigger="FORCED",
                orchestration_mode=self.orchestration_mode.value,
            ))

        return tick_logs

    # ──────────────────────────────────────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_tick_logs(
        self,
        tick_results: list[dict[str, Any]],
        tick: int,
        trigger: str = "HOMEOSTATIC",
    ) -> list[SimulationLog]:
        """Convert raw agent tick results into SimulationLog entries.

        Parameters
        ----------
        trigger:
            Activation trigger for all entries in this batch.
            "HOMEOSTATIC" for HRRL (sigmoid fired), "ROUTER" for LangGraph.
            PASS entries always carry the same trigger as the batch — it
            indicates the mechanism that *evaluated* the agent, even if it
            chose not to act.
        """
        logs: list[SimulationLog] = []
        summary = self.graph.get_state_summary()

        for res in tick_results:
            if res["action_type"] == "DEBATE" and res["node_id"]:
                self._total_debate_turns += 1
                if res.get("target_node_id") and self.graph.graph.has_node(res["target_node_id"]):
                    self._consistent_turns += 1

            utility = res.get("utility_scores", {})
            # Extract best DEBATE utility from stimulus pool
            debate_utils = [v for k, v in utility.items() if k.startswith("DEBATE")]
            best_debate_u = max(debate_utils) if debate_utils else utility.get("DEBATE", 0.0)

            logs.append(SimulationLog(
                tick=tick,
                agent_id=res["agent_id"],
                action=res["action_type"],
                claim=res.get("claim"),
                target_node_id=res.get("target_node_id"),
                attack_type=res.get("attack_type"),
                deficit_before=round(res["deficit_before"], 4),
                deficit_after=round(res["deficit_after"], 4),
                delta_phi=round(res["delta_phi"], 4),
                activation_prob=round(res["activation_prob"], 4),
                graph_nodes=summary["nodes"],
                graph_edges=summary["edges"],
                consistency_rate=round(self.consistency_rate, 4),
                trigger=trigger,
                utility_debate=best_debate_u,
                utility_search=utility.get("SEARCH", 0.0),
                utility_read=utility.get("READ", 0.0),
                utility_pass=utility.get("PASS", 0.0),
                orchestration_mode=self.orchestration_mode.value,
                stimulus_event_id=res.get("stimulus_event_id"),
                stimulus_utility=round(res.get("stimulus_utility", 0.0), 4),
                n_stimuli_evaluated=res.get("n_stimuli_evaluated", 0),
                reward=round(res.get("reward", 0.0), 6),
                q_values=res.get("q_values", {}),
            ))

        return logs

    def _print_tick(self, tick: int, tick_logs: list[SimulationLog]) -> None:
        table = Table(
            title=f"Tick {tick} [{self.orchestration_mode.value}]",
            show_lines=True,
            title_style="bold green",
        )
        table.add_column("Agent", style="cyan", width=8)
        table.add_column("Action", width=8)
        table.add_column("Deficit", justify="right", width=10)
        table.add_column("Δφ*", justify="right", width=8)
        table.add_column("p(act)", justify="right", width=8)
        table.add_column("Claim", max_width=50, overflow="ellipsis")

        for entry in tick_logs:
            action_style = "bold red" if entry.action == "DEBATE" else (
                "bold blue" if entry.action in ("SEARCH", "READ") else "dim"
            )
            table.add_row(
                entry.agent_id,
                f"[{action_style}]{entry.action}[/{action_style}]",
                f"{entry.deficit_after:.4f}",
                f"{entry.delta_phi:.4f}",
                f"{entry.activation_prob:.4f}",
                (entry.claim or "-")[:50],
            )

        summary = self.graph.get_state_summary()
        console.print(table)
        console.print(
            f"  Graph: [bold]{summary['nodes']}[/bold] nodes, "
            f"[bold]{summary['edges']}[/bold] edges, "
            f"density={summary['density']:.4f}  |  "
            f"consistency={self.consistency_rate:.4f}  |  "
            f"API calls: {self.budget.summary()}\n"
        )

    def _print_summary(self) -> None:
        summary = self.graph.get_state_summary()
        console.print("\n[bold yellow]═══ Final Summary ═══[/bold yellow]")
        console.print(f"  Mode             : {self.orchestration_mode.value}")
        console.print(f"  Total ticks      : {self.max_iterations}")
        console.print(f"  Debate turns     : {self._total_debate_turns}")
        console.print(f"  Consistent turns : {self._consistent_turns}")
        console.print(f"  Consistency rate : {self.consistency_rate:.4f}")
        console.print(f"  Graph nodes      : {summary['nodes']}")
        console.print(f"  Graph edges      : {summary['edges']}")
        console.print(f"  Graph density    : {summary['density']:.4f}")
        console.print(f"  API calls total  : {self.budget.summary()}")
        console.print("\n  [bold]Agent Deficits:[/bold]")
        for agent in self.agents:
            console.print(f"    {agent.agent_id}: {agent.drive.deficit:.4f}")
