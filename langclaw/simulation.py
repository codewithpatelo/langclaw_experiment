"""Sotopia zero-sum debate environment with VSM-structured agents.

Three operating modes controlled by ``orchestration_mode``:

  hrrl       — Agents run as asyncio coroutines; activation is endogenous
               (homeostatic sigmoid gate + cognitive loop). Each agent
               independently decides whether to participate via its
               epistemic drive. DirectMessages routed per-recipient.
               Trigger tag: HOMEOSTATIC.

  langgraph  — A neutral LLM router reads current discourse state and selects
               the next speaker each tick (exogenous, state-informed routing).
               No directed messaging. Trigger tag: ROUTER.

  round-robin — Every agent speaks in fixed order every tick. Trigger: FORCED.
  random      — A randomly selected agent speaks each tick. Trigger: FORCED.

Agent structure: 10 agents organized as two viable systems (Beer's VSM),
  5 per faction (S1 Operations through S5 Strategy).
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
from langclaw.events import (
    DirectMessageEvent,
    NewArgumentEvent,
    SimulationEndEvent,
    TickElapsedEvent,
)
from langclaw.router import LangGraphRouter
from langclaw.schemas import AgentState, SimulationLog

logger = logging.getLogger(__name__)
console = Console()


class OrchestrationMode(str, Enum):
    HRRL = "hrrl"
    LANGGRAPH = "langgraph"
    ROUND_ROBIN = "round-robin"
    RANDOM = "random"


# ──────────────────────────────────────────────────────────────────────────────
# VSM Agent Roles: 5 subsystems × 2 factions = 10 agents
# ──────────────────────────────────────────────────────────────────────────────

AGENT_ROLES: list[dict[str, str]] = [
    # ── Government faction ──
    {
        "id": "GOV-S1",
        "vsm_system": "S1",
        "archetype": "Operations",
        "prompt": (
            "Eres el agente operativo (Sistema 1 - Operaciones) de la facción gobierno. "
            "Tu función principal es producir argumentos: atacar las posiciones de la "
            "oposición y defender las del gobierno usando datos, evidencia y razonamiento "
            "lógico. Eres el debatidor principal. Ejecutas las directivas que recibas de "
            "tus compañeros de facción. Argumenta en español."
        ),
    },
    {
        "id": "GOV-S2",
        "vsm_system": "S2",
        "archetype": "Coordination",
        "prompt": (
            "Eres el agente coordinador (Sistema 2 - Coordinación) de la facción gobierno. "
            "Tu función es coordinar las acciones de tu facción: delegar tareas a otros "
            "agentes, transmitir las directivas de S5 hacia los demás, asignar recursos "
            "argumentativos y sincronizar timing para evitar que dos agentes de la misma "
            "facción ataquen el mismo nodo. Puedes debatir pero tu valor principal es "
            "coordinar. Usa MESSAGE para comunicarte con tu facción. Argumenta en español."
        ),
    },
    {
        "id": "GOV-S3",
        "vsm_system": "S3",
        "archetype": "Control",
        "prompt": (
            "Eres el agente de control interno (Sistema 3 - Control) de la facción gobierno. "
            "Tu función es auditar la coherencia interna de tu facción: detectar "
            "contradicciones entre los argumentos de tus compañeros, corregir "
            "inconsistencias y asegurar que la línea argumental sea coherente. Si detectas "
            "que un compañero contradice la posición de la facción, usa MESSAGE para "
            "informarle. Puedes debatir pero tu prioridad es el control interno. "
            "Argumenta en español."
        ),
    },
    {
        "id": "GOV-S4",
        "vsm_system": "S4",
        "archetype": "Intelligence",
        "prompt": (
            "Eres el agente de inteligencia (Sistema 4 - Inteligencia) de la facción gobierno. "
            "Tu función es mirar hacia afuera y hacia adelante: anticipar las amenazas del "
            "oponente, escanear patrones en sus argumentos, identificar debilidades "
            "emergentes e investigar recursos (usa SEARCH para obtener datos). Informa a "
            "S5 sobre lo que descubras vía MESSAGE. Tu valor es la previsión estratégica. "
            "Argumenta en español."
        ),
    },
    {
        "id": "GOV-S5",
        "vsm_system": "S5",
        "archetype": "Strategy",
        "prompt": (
            "Eres el agente estratega (Sistema 5 - Estrategia) de la facción gobierno. "
            "Tu función es definir la estrategia de la facción: establecer prioridades, "
            "resolver tensiones entre S3 (control) y S4 (inteligencia), y enviar "
            "directivas estratégicas a S1-S4 vía MESSAGE. Decides qué claims atacar "
            "primero, cuándo cambiar de táctica y qué línea argumental priorizar. "
            "Puedes debatir pero tu valor principal es la dirección estratégica. "
            "Argumenta en español."
        ),
    },
    # ── Opposition faction ──
    {
        "id": "OPP-S1",
        "vsm_system": "S1",
        "archetype": "Operations",
        "prompt": (
            "Eres el agente operativo (Sistema 1 - Operaciones) de la facción oposición. "
            "Tu función principal es producir argumentos: atacar las posiciones del "
            "gobierno y defender las de la oposición usando datos, evidencia y "
            "razonamiento lógico. Eres el debatidor principal. Ejecutas las directivas "
            "que recibas de tus compañeros de facción. Argumenta en español."
        ),
    },
    {
        "id": "OPP-S2",
        "vsm_system": "S2",
        "archetype": "Coordination",
        "prompt": (
            "Eres el agente coordinador (Sistema 2 - Coordinación) de la facción oposición. "
            "Tu función es coordinar las acciones de tu facción: delegar tareas, "
            "transmitir directivas de S5, asignar recursos argumentativos y sincronizar "
            "timing. Usa MESSAGE para comunicarte con tu facción. Argumenta en español."
        ),
    },
    {
        "id": "OPP-S3",
        "vsm_system": "S3",
        "archetype": "Control",
        "prompt": (
            "Eres el agente de control interno (Sistema 3 - Control) de la facción oposición. "
            "Tu función es auditar la coherencia interna: detectar contradicciones entre "
            "los argumentos de tus compañeros, corregir inconsistencias y asegurar "
            "coherencia. Usa MESSAGE para informar sobre problemas. Argumenta en español."
        ),
    },
    {
        "id": "OPP-S4",
        "vsm_system": "S4",
        "archetype": "Intelligence",
        "prompt": (
            "Eres el agente de inteligencia (Sistema 4 - Inteligencia) de la facción oposición. "
            "Tu función es mirar hacia afuera: anticipar amenazas del gobierno, escanear "
            "patrones, identificar debilidades e investigar recursos (SEARCH). Informa a "
            "S5 vía MESSAGE. Argumenta en español."
        ),
    },
    {
        "id": "OPP-S5",
        "vsm_system": "S5",
        "archetype": "Strategy",
        "prompt": (
            "Eres el agente estratega (Sistema 5 - Estrategia) de la facción oposición. "
            "Tu función es definir la estrategia: establecer prioridades, resolver "
            "tensiones entre S3 y S4, y enviar directivas estratégicas a S1-S4 vía "
            "MESSAGE. Decides qué claims atacar primero y qué línea priorizar. "
            "Argumenta en español."
        ),
    },
]


def _faction_of(agent_id: str) -> str:
    """Extract faction prefix from agent ID (e.g. 'GOV-S1' -> 'GOV')."""
    return agent_id.split("-")[0] if "-" in agent_id else agent_id


def _faction_agents(agent_id: str) -> list[str]:
    """Return the list of agent IDs in the same faction."""
    faction = _faction_of(agent_id)
    return [r["id"] for r in AGENT_ROLES if _faction_of(r["id"]) == faction]


class SotopiaEnvironment:
    """Zero-sum political-survival debate environment with VSM agents."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3",
        api_key: str = "ollama",
        max_iterations: int = 80,
        seed: int | None = None,
        orchestration_mode: OrchestrationMode | str = OrchestrationMode.HRRL,
        api_hard_limit: int = 500,
        tick_interval: float = 0.0,
        initial_deficit: float = 0.5,
        on_tick: Callable[[int, list[SimulationLog], "SotopiaEnvironment"], None] | None = None,
        max_debates: int | None = None,
        stimulus_weights: dict[str, float] | None = None,
        debate_alpha: float = 2.0,
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
                vsm_system=role.get("vsm_system"),
                faction_agents=_faction_agents(role["id"]),
                stimulus_weights=stimulus_weights,
                debate_alpha=debate_alpha,
            )
            for role in AGENT_ROLES
        ]

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
        if self._total_debate_turns == 0:
            return 0.0
        return self._consistent_turns / self._total_debate_turns

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry points
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> list[SimulationLog]:
        """Execute the full simulation and return the event log."""
        console.rule(
            f"[bold cyan]LangClaw Simulation - mode: {self.orchestration_mode.value} "
            f"({len(self.agents)} agents)[/bold cyan]"
        )
        if self._seed_factory:
            console.print("  [dim]Seeds derivadas (master -> primos):[/dim]")
            for component, prime in self._seed_factory.summary().items():
                console.print(f"  [dim]  {component:<30} -> {prime}[/dim]")

        if self.orchestration_mode == OrchestrationMode.HRRL:
            first_agent = self.agents[0]
            d0 = first_agent.drive.deficit
            theta = 0.7
            lambda_rate = 0.05
            ticks_to_50pct = max(0.0, (theta - d0) / lambda_rate)
            console.print(
                f"  [dim]Initial deficit: {d0:.2f} | theta={theta} | lambda={lambda_rate}/tick "
                f"-> 50% activation at tick ~{ticks_to_50pct:.0f}[/dim]"
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
        if self.orchestration_mode == OrchestrationMode.HRRL:
            tick_logs = asyncio.run(self._hrrl_single_tick(tick))
        elif self.orchestration_mode == OrchestrationMode.LANGGRAPH:
            saved_max = self.max_iterations
            self.max_iterations = 1
            self._run_langgraph()
            self.max_iterations = saved_max
            tick_logs = [l for l in self.logs if l.tick == tick]
            return tick_logs
        else:
            tick_logs = self._baseline_tick(tick)
        self.logs.extend(tick_logs)
        return tick_logs

    # ──────────────────────────────────────────────────────────────────────────
    # HRRL async loop with directed messaging
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_hrrl(self) -> None:
        agent_queues: dict[str, asyncio.Queue] = {
            a.agent_id: asyncio.Queue() for a in self.agents
        }
        output_queue: asyncio.Queue = asyncio.Queue()
        llm_semaphore = asyncio.Semaphore(2)

        agent_tasks = [
            asyncio.create_task(
                agent.run(
                    agent_queues[agent.agent_id], self.graph, self.budget,
                    output_queue, llm_semaphore=llm_semaphore,
                ),
                name=agent.agent_id,
            )
            for agent in self.agents
        ]

        n_agents = len(self.agents)

        for tick in range(1, self.max_iterations + 1):
            for q in agent_queues.values():
                await q.put(TickElapsedEvent(tick=tick))

            tick_results: list[dict[str, Any]] = []
            for _ in range(n_agents):
                result = await output_queue.get()
                tick_results.append(result)
                output_queue.task_done()

            # Broadcast NewArgumentEvents (public discourse)
            new_arg_events: list[NewArgumentEvent] = []
            for res in tick_results:
                if res["action_type"] == "DEBATE" and res.get("node_id"):
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

            # Route DirectMessages to specific recipients
            for res in tick_results:
                for msg in res.get("messages", []):
                    to_agent = msg.get("to_agent", "")
                    if to_agent in agent_queues:
                        dm = DirectMessageEvent(
                            tick=tick,
                            from_agent=res["agent_id"],
                            to_agent=to_agent,
                            content=msg.get("content", ""),
                            performative=msg.get("msg_type", "inform"),
                        )
                        await agent_queues[to_agent].put(dm)

            tick_logs = self._build_tick_logs(tick_results, tick)
            self.logs.extend(tick_logs)
            self._print_tick(tick, tick_logs)

            if self._on_tick:
                self._on_tick(tick, tick_logs, self)

            if self.tick_interval > 0:
                await asyncio.sleep(self.tick_interval)

        for q in agent_queues.values():
            await q.put(SimulationEndEvent())
        await asyncio.gather(*agent_tasks)

    async def _hrrl_single_tick(self, tick: int) -> list[SimulationLog]:
        agent_queues: dict[str, asyncio.Queue] = {
            a.agent_id: asyncio.Queue() for a in self.agents
        }
        output_queue: asyncio.Queue = asyncio.Queue()
        llm_semaphore = asyncio.Semaphore(2)

        agent_tasks = [
            asyncio.create_task(
                agent.run(
                    agent_queues[agent.agent_id], self.graph, self.budget,
                    output_queue, llm_semaphore=llm_semaphore,
                ),
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
        """LangGraph compiled-graph orchestration.

        Same agents, same action space, same memory, same cognitive phases.
        The cognitive loop (THINK → PLAN → EXECUTE → OBSERVE) is materialized
        as a compiled LangGraph StateGraph. The graph's conditional edges
        control the flow — not the agent's sigmoid or Q-learner.

        Each tick:
          1. Router selects which agent enters the graph.
          2. ALL agents experience heartbeat decay (time passes for everyone).
          3. Selected agent's cognitive loop runs through the compiled graph.
          4. If graph says EXECUTE → agent calls LLM with same prompt/capabilities.
          5. DirectMessages are routed (same infrastructure as HRRL).
          6. Non-selected agents buffer events for next tick.
        """
        from langclaw.langgraph_flow import build_cognitive_graph

        assert self._router is not None, "Router not initialised for LANGGRAPH mode"
        agent_ids = [a.agent_id for a in self.agents]
        agent_map = {a.agent_id: a for a in self.agents}

        cognitive_graph = build_cognitive_graph()

        pending_messages: dict[str, list[DirectMessageEvent]] = {
            a.agent_id: [] for a in self.agents
        }

        for tick in range(1, self.max_iterations + 1):
            discourse_context = self.graph.get_recent_context(last_n=6)
            selected_id = self._router.select_next_agent(
                discourse_context=discourse_context,
                agent_ids=agent_ids,
            )

            tick_results: list[dict[str, Any]] = []

            for agent in self.agents:
                deficit_before = agent.drive.deficit
                agent.drive.decay()
                agent.memory.update_working_tick(tick)

                # Drain buffered events for stimulus
                buffered_events = list(agent._event_buffer)
                agent._event_buffer.clear()
                for evt in buffered_events:
                    relevance = agent.stimulus_evaluator.evaluate(
                        evt, agent.agent_id, agent.memory, self.graph
                    )
                    agent.drive.stimulate(relevance, gamma=0.1)

                incoming_msgs = list(pending_messages[agent.agent_id])
                pending_messages[agent.agent_id].clear()

                activation_prob = agent.drive.get_activation_probability()

                res: dict[str, Any] = {
                    "agent_id": agent.agent_id,
                    "tick": tick,
                    "action_type": "PASS",
                    "node_id": None,
                    "claim": None,
                    "target_node_id": None,
                    "attack_type": None,
                    "deficit_before": deficit_before,
                    "deficit_after": agent.drive.deficit,
                    "activation_prob": activation_prob,
                    "delta_phi": 0.0,
                    "utility_scores": {},
                    "stimulus_event_id": None,
                    "stimulus_utility": 0.0,
                    "n_stimuli_evaluated": len(buffered_events),
                    "n_messages_received": len(incoming_msgs),
                    "reward": 0.0,
                    "q_values": {},
                    "agent_state": "active",
                    "cognitive_phase": None,
                    "messages": [],
                    "vsm_system": agent.vsm_system,
                    "send_to": None,
                    "message_content": None,
                    "message_type": None,
                }

                if agent.agent_id != selected_id:
                    tick_results.append(res)
                    continue

                # Run the compiled cognitive graph for the selected agent
                messages_ctx = "No messages."
                if incoming_msgs:
                    msg_lines = [
                        f"[{m.performative.upper()}] from {m.from_agent}: {m.content}"
                        for m in incoming_msgs
                    ]
                    messages_ctx = "\n".join(msg_lines)

                faction_agents_str = ", ".join(
                    a for a in agent.faction_agents if a != agent.agent_id
                ) or "None"

                target_ids = self.graph.valid_target_ids()
                discourse_query = discourse_context[:200] if discourse_context else None
                memory_ctx = agent.memory.get_prompt_context(
                    discourse_query=discourse_query
                )

                best_stimulus_ctx = "No specific stimulus. Act proactively."
                if buffered_events:
                    scored = [
                        (e, agent.stimulus_evaluator.evaluate(
                            e, agent.agent_id, agent.memory, self.graph
                        ))
                        for e in buffered_events
                    ]
                    best_evt, best_u = max(scored, key=lambda x: x[1])
                    res["stimulus_event_id"] = best_evt.node_id
                    res["stimulus_utility"] = best_u
                    best_stimulus_ctx = (
                        f"You are responding to: [{best_evt.node_id}] "
                        f"{best_evt.agent_id} ({best_evt.faction}): "
                        f'"{best_evt.claim}"'
                    )

                graph_state = {
                    "agent_id": agent.agent_id,
                    "tick": tick,
                    "deficit": agent.drive.deficit,
                    "graph_context": discourse_context or "",
                    "target_ids": target_ids,
                    "faction_agents": faction_agents_str,
                    "memory_context": memory_ctx,
                    "messages_context": messages_ctx,
                    "stimulus_context": best_stimulus_ctx,
                    "role_prompt": agent.role_prompt,
                    "action": "PASS",
                    "claim": None,
                    "target_node_id": None,
                    "attack_type": None,
                    "send_to": None,
                    "message_content": None,
                    "message_type": None,
                    "delta_phi": 0.0,
                    "phase": "triage",
                    "should_act": False,
                    "llm_response": None,
                    "budget_ok": self.budget.can_call(
                        agent.agent_id, agent.drive.deficit, tick
                    ),
                }

                # Invoke the compiled graph
                graph_result = cognitive_graph.invoke(graph_state)
                res["cognitive_phase"] = graph_result.get("phase", "pass")
                res["agent_state"] = "working"

                # If graph decided to act (phase == "execute" or "observe")
                if graph_result.get("phase") in ("execute", "observe"):
                    faction_prefix = agent.agent_id.split("-")[0] + "-"
                    undefended = self.graph.get_undefended_attacks(faction_prefix)
                    undef_ctx = "None — your faction's claims are all defended."
                    if undefended:
                        udef_lines = []
                        for u in undefended[:5]:
                            udef_lines.append(
                                f"- [{u['attacker_node']}] attacks your "
                                f"[{u['attacked_node']}]: \"{u['attacker_claim']}\""
                            )
                        undef_ctx = "\n".join(udef_lines)

                    action_result = agent.step(
                        discourse_context,
                        target_ids,
                        incoming_messages=incoming_msgs,
                        stimulus_context=best_stimulus_ctx,
                        undefended_attacks_ctx=undef_ctx,
                    )

                    if action_result and action_result.action == "DEBATE" and action_result.claim:
                        node_id = self.graph.add_argument(
                            agent_id=agent.agent_id,
                            claim=action_result.claim,
                            target_node_id=action_result.target_node_id,
                            attack_type=action_result.attack_type,
                            tick=tick,
                        )
                        delta_phi = self.graph.calculate_phi_star_proxy(node_id)
                        agent.learn(discourse_context, action_result, delta_phi)
                        res.update({
                            "action_type": "DEBATE",
                            "node_id": node_id,
                            "claim": action_result.claim,
                            "target_node_id": action_result.target_node_id,
                            "attack_type": action_result.attack_type,
                            "delta_phi": delta_phi,
                        })

                    elif action_result and action_result.action == "MESSAGE" and action_result.send_to:
                        res.update({
                            "action_type": "MESSAGE",
                            "send_to": action_result.send_to,
                            "message_content": action_result.message_content,
                            "message_type": action_result.message_type,
                        })
                        res["messages"] = [{
                            "to_agent": action_result.send_to,
                            "content": action_result.message_content or "",
                            "msg_type": action_result.message_type or "inform",
                        }]
                        content = (action_result.message_content or "").strip()
                        has_recipient = action_result.send_to in agent.faction_agents
                        word_count = len(content.split()) if content else 0
                        substance = min(1.0, word_count / 20.0) if has_recipient else 0.0
                        agent.drive.satiate(substance * 0.08, alpha=1.0)

                    elif action_result and action_result.action in ("SEARCH", "READ"):
                        res["action_type"] = action_result.action

                res["deficit_after"] = agent.drive.deficit

                # Compute reward (same as HRRL, for logging parity)
                reward = agent.drive.compute_reward(
                    delta_before=deficit_before,
                    delta_after=agent.drive.deficit,
                    epsilon=agent.drive.BASELINE,
                    m=agent.drive.m,
                )
                res["reward"] = round(reward, 6)

                tick_results.append(res)

            # Broadcast NewArgumentEvents (same as HRRL)
            for r in tick_results:
                if r["action_type"] == "DEBATE" and r.get("node_id"):
                    author_faction = _faction_of(r["agent_id"])
                    target_faction: str | None = None
                    target_nid = r.get("target_node_id")
                    if target_nid and self.graph.graph.has_node(target_nid):
                        ta = self.graph.graph.nodes[target_nid].get("agent_id", "")
                        target_faction = _faction_of(ta)
                    evt = NewArgumentEvent(
                        tick=tick,
                        node_id=r["node_id"],
                        agent_id=r["agent_id"],
                        claim=r["claim"] or "",
                        delta_phi=r["delta_phi"],
                        attack_type=r.get("attack_type"),
                        target_node_id=target_nid,
                        faction=author_faction,
                        targets_faction=target_faction,
                    )
                    for agent in self.agents:
                        if agent.agent_id != r["agent_id"]:
                            agent.memory.observe(evt)
                            agent._event_buffer.append(evt)

            # Route DirectMessages (same as HRRL)
            for r in tick_results:
                for msg in r.get("messages", []):
                    to_id = msg.get("to_agent", "")
                    if to_id in pending_messages:
                        pending_messages[to_id].append(DirectMessageEvent(
                            tick=tick,
                            from_agent=r["agent_id"],
                            to_agent=to_id,
                            content=msg.get("content", ""),
                            performative=msg.get("msg_type", "inform"),
                        ))

            tick_logs = self._build_tick_logs(tick_results, tick, trigger="ROUTER")
            self.logs.extend(tick_logs)
            self._print_tick(tick, tick_logs)

            if self._on_tick:
                self._on_tick(tick, tick_logs, self)

    def _baseline_tick(self, tick: int) -> list[SimulationLog]:
        if self.orchestration_mode == OrchestrationMode.ROUND_ROBIN:
            acting_agents = self.agents
        else:
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
                faction_prefix = agent.agent_id.split("-")[0] + "-"
                undefended = self.graph.get_undefended_attacks(faction_prefix)
                undef_ctx = "None — your faction's claims are all defended."
                if undefended:
                    udef_lines = []
                    for u in undefended[:5]:
                        udef_lines.append(
                            f"- [{u['attacker_node']}] attacks your "
                            f"[{u['attacked_node']}]: \"{u['attacker_claim']}\""
                        )
                    undef_ctx = "\n".join(udef_lines)
                result = agent.step(graph_context, target_ids, undefended_attacks_ctx=undef_ctx)

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
                vsm_system=agent.vsm_system,
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
        logs: list[SimulationLog] = []
        summary = self.graph.get_state_summary()

        for res in tick_results:
            if res["action_type"] == "DEBATE" and res.get("node_id"):
                self._total_debate_turns += 1
                if res.get("target_node_id") and self.graph.graph.has_node(res["target_node_id"]):
                    self._consistent_turns += 1

            utility = res.get("utility_scores", {})
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
                agent_state=res.get("agent_state", "active"),
                cognitive_phase=res.get("cognitive_phase"),
                send_to=res.get("send_to"),
                message_content=res.get("message_content"),
                message_type=res.get("message_type"),
                n_messages_received=res.get("n_messages_received", 0),
                vsm_system=res.get("vsm_system"),
            ))

        return logs

    def _print_tick(self, tick: int, tick_logs: list[SimulationLog]) -> None:
        table = Table(
            title=f"Tick {tick} [{self.orchestration_mode.value}]",
            show_lines=True,
            title_style="bold green",
        )
        table.add_column("Agent", style="cyan", width=8)
        table.add_column("VSM", width=4)
        table.add_column("Action", width=8)
        table.add_column("Deficit", justify="right", width=10)
        table.add_column("Δφ*", justify="right", width=8)
        table.add_column("p(act)", justify="right", width=8)
        table.add_column("Claim/Message", max_width=40, overflow="ellipsis")

        for entry in tick_logs:
            action_style = "bold red" if entry.action == "DEBATE" else (
                "bold blue" if entry.action in ("SEARCH", "READ") else (
                    "bold magenta" if entry.action == "MESSAGE" else "dim"
                )
            )
            display_text = entry.claim or entry.message_content or "-"
            if entry.action == "MESSAGE" and entry.send_to:
                display_text = f"→{entry.send_to}: {display_text}"
            table.add_row(
                entry.agent_id,
                entry.vsm_system or "-",
                f"[{action_style}]{entry.action}[/{action_style}]",
                f"{entry.deficit_after:.4f}",
                f"{entry.delta_phi:.4f}",
                f"{entry.activation_prob:.4f}",
                (display_text or "-")[:40],
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
        console.print(f"  Agents           : {len(self.agents)}")
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
            console.print(
                f"    {agent.agent_id} ({agent.vsm_system or '?'}): "
                f"{agent.drive.deficit:.4f}"
            )
