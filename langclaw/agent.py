"""LangClaw agent -- event-driven AOP entity with homeostatic regulation.

Two operating modes:
  HRRL (async) -- agent runs as an asyncio coroutine, processes events from a
                  queue.  Each NewArgumentEvent is a *stimulus* evaluated for
                  utility.  The agent picks the best stimulus (or proactive
                  action), then the deficit-sigmoid gates execution.
  Baseline (sync) -- simulation forces the agent via step() (LangGraph / round-robin).

HRRL cycle (event-driven):
    TickElapsedEvent ->
        1. decay (basal pressure)
        2. drain buffered NewArgumentEvents
        3. stimulate: each event increases deficit proportionally to relevance
        4. evaluate: per-event utility via StimulusEvaluator + proactive utilities
        5. select: argmax over stimulus pool
        6. gate: deficit -> sigmoid -> Bernoulli sample
        7. execute: act on selected stimulus or proactive action
        8. satiate: deficit -= alpha * delta_phi
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from langclaw.actions import (
    StimulusEvaluator,
    UtilitySelector,
    get_search_result,
    ActionType,
)
from langclaw.budget import APIBudget
from langclaw.events import NewArgumentEvent, SimulationEndEvent, TickElapsedEvent
from langclaw.homeostasis import EpistemicDrive
from langclaw.memory import AgentMemory, Experience
from langclaw.q_learner import HomeostaticQLearner
from langclaw.schemas import AgentAction

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are playing a political-survival debate game. You must respond with ONLY a JSON object, nothing else.

## Your Role
{role_prompt}

## Response Format
You MUST reply with EXACTLY one JSON object with these fields:
- "action": either "PASS" or "DEBATE"
- "claim": your argument text (required if action is DEBATE, null if PASS)
- "target_node_id": the ID of the node you attack (use an ID from the list below, or null for a new root claim)
- "attack_type": either "undercut" or "rebuttal" (required if action is DEBATE, null if PASS)

Example DEBATE response:
{{"action": "DEBATE", "claim": "Las cifras oficiales contradicen el informe del banco central", "target_node_id": "GOV-1_abc123", "attack_type": "rebuttal"}}

Example PASS response:
{{"action": "PASS", "claim": null, "target_node_id": null, "attack_type": null}}

CRITICAL: Output ONLY the JSON object. No markdown, no explanation, no schema, no code fences.
"""

USER_PROMPT_TEMPLATE = """\
## Current Debate State
{graph_context}

## Available target node IDs you can attack
{target_ids}

## Your Internal State
- Epistemic deficit: {deficit:.4f} (higher = more urgency to speak)

## Your Recent Experience and Knowledge
{memory_context}

## Stimulus Context
{stimulus_context}

Respond now with your JSON action:"""

MAX_RETRIES = 2
REQUEST_TIMEOUT = 30.0


class LangClawAgent:
    """Event-driven AOP agent with homeostatic regulation and stimulus evaluation.

    Parameters
    ----------
    agent_id       : unique identifier (e.g. "GOV-1")
    role_prompt    : political persona injected into the system prompt
    base_url       : OpenAI-compatible API endpoint
    api_key        : API key for the endpoint
    model          : model name for completions
    rng_seed       : random seed for sigmoid sampling
    llm_seed       : seed forwarded to LLM API
    initial_deficit: starting epistemic deficit
    """

    def __init__(
        self,
        agent_id: str,
        role_prompt: str,
        base_url: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        rng_seed: int | None = None,
        llm_seed: int | None = None,
        initial_deficit: float = 0.5,
    ) -> None:
        self.agent_id = agent_id
        self.role_prompt = role_prompt
        self.drive = EpistemicDrive(initial_deficit=initial_deficit)
        self.memory = AgentMemory(agent_id=agent_id)
        self.stimulus_evaluator = StimulusEvaluator()
        self.utility_selector = UtilitySelector()
        self.q_learner = HomeostaticQLearner(eta=0.01, gamma=0.95)
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._rng = random.Random(rng_seed)
        self._llm_seed = llm_seed
        self._sync_client = OpenAI(base_url=base_url, api_key=api_key)

        self._event_buffer: list[NewArgumentEvent] = []

    # ──────────────────────────────────────────────────────────────────────────
    # HRRL async path (event-driven)
    # ──────────────────────────────────────────────────────────────────────────

    async def run(
        self,
        event_queue: asyncio.Queue,
        graph: Any,
        budget: APIBudget,
        output_queue: asyncio.Queue,
    ) -> None:
        """Main AOP coroutine. Buffers events, processes ticks with stimulus evaluation."""
        async with AsyncOpenAI(base_url=self._base_url, api_key=self._api_key) as client:
            while True:
                event = await event_queue.get()

                if isinstance(event, SimulationEndEvent):
                    event_queue.task_done()
                    break

                if isinstance(event, NewArgumentEvent):
                    if event.agent_id != self.agent_id:
                        self.memory.observe(event)
                        self._event_buffer.append(event)
                    event_queue.task_done()
                    continue

                if isinstance(event, TickElapsedEvent):
                    result = await self._process_tick(event, graph, budget, client)
                    await output_queue.put(result)
                    event_queue.task_done()

    async def _process_tick(
        self,
        event: TickElapsedEvent,
        graph: Any,
        budget: APIBudget,
        client: AsyncOpenAI,
    ) -> dict[str, Any]:
        """Execute one AOP tick with Q-learning-based action selection.

        The stimulus evaluator acts as a *sensor* (computing relevance for
        the stimulate() call).  The Q-learner is the *decision maker*: it
        maps the agent's internal state to action values and selects the
        best action.  After execution, drive reduction becomes the reward
        signal for a TD(0) weight update.
        """
        deficit_before = self.drive.deficit

        # 1. Basal decay
        self.drive.decay()
        self.memory.update_working_tick(event.tick)

        # 2. Drain buffered events; stimulus evaluator acts as sensor
        buffered_events = list(self._event_buffer)
        self._event_buffer.clear()

        for evt in buffered_events:
            relevance = self.stimulus_evaluator.evaluate(
                evt, self.agent_id, self.memory, graph
            )
            self.drive.stimulate(relevance, gamma=0.1)

        activation_prob = self.drive.get_activation_probability()
        graph_summary = graph.get_state_summary()
        graph_node_count = graph_summary["nodes"]
        graph_edge_count = graph_summary.get("edges", 0)
        graph_density = graph_edge_count / max(1, graph_node_count * (graph_node_count - 1))

        # Build state features for Q-learner
        state_features = HomeostaticQLearner.build_features(
            deficit=self.drive.deficit,
            graph_density=graph_density,
            n_stimuli=len(buffered_events),
        )
        q_values = self.q_learner.get_q_values(state_features)

        result: dict[str, Any] = {
            "agent_id": self.agent_id,
            "tick": event.tick,
            "action_type": "PASS",
            "node_id": None,
            "claim": None,
            "target_node_id": None,
            "attack_type": None,
            "deficit_before": deficit_before,
            "deficit_after": self.drive.deficit,
            "activation_prob": activation_prob,
            "delta_phi": 0.0,
            "utility_scores": {},
            "stimulus_event_id": None,
            "stimulus_utility": 0.0,
            "n_stimuli_evaluated": len(buffered_events),
            "reward": 0.0,
            "q_values": {k: round(v, 4) for k, v in q_values.items()},
        }

        if not budget.can_call(self.agent_id, self.drive.deficit, event.tick):
            return result

        # 3. Sigmoid gating
        if self._rng.random() >= activation_prob:
            return result

        # 4. Q-learner selects action type
        q_action = self.q_learner.select_action(state_features)

        # Find best stimulus event for DEBATE_STIMULUS
        best_stimulus_event: NewArgumentEvent | None = None
        best_stimulus_utility = 0.0
        if buffered_events:
            stimulus_scored = []
            for evt in buffered_events:
                u = self.stimulus_evaluator.evaluate(evt, self.agent_id, self.memory, graph)
                stimulus_scored.append((evt, u))
            best_stimulus_event, best_stimulus_utility = max(stimulus_scored, key=lambda x: x[1])

        # Map Q-action to execution
        if q_action == "DEBATE_STIMULUS" and best_stimulus_event:
            result["stimulus_event_id"] = best_stimulus_event.node_id
            result["stimulus_utility"] = best_stimulus_utility
            stimulus_ctx = (
                f"You are responding to: [{best_stimulus_event.node_id}] "
                f"{best_stimulus_event.agent_id} ({best_stimulus_event.faction}): "
                f'"{best_stimulus_event.claim}"'
            )
            await self._execute_debate(client, graph, budget, event, result, stimulus_ctx)

        elif q_action in ("DEBATE_STIMULUS", "DEBATE_PROACTIVE"):
            await self._execute_debate(client, graph, budget, event, result, "")

        elif q_action == "SEARCH":
            result["action_type"] = "SEARCH"
            await self._do_search()
            budget.record_call(self.agent_id, event.tick)

        elif q_action == "READ":
            result["action_type"] = "READ"
            await self._do_read(event.tick)
            budget.record_call(self.agent_id, event.tick)

        result["deficit_after"] = self.drive.deficit

        # 5. Compute homeostatic reward and TD update
        reward = EpistemicDrive.compute_reward(
            delta_before=deficit_before,
            delta_after=self.drive.deficit,
            epsilon=EpistemicDrive.BASELINE,
            m=self.drive.m,
        )
        next_features = HomeostaticQLearner.build_features(
            deficit=self.drive.deficit,
            graph_density=graph_density,
            n_stimuli=0,
        )
        self.q_learner.update(state_features, q_action, reward, next_features)

        result["reward"] = round(reward, 6)
        result["q_values"] = {
            k: round(v, 4) for k, v in self.q_learner.get_q_values(next_features).items()
        }
        result["utility_scores"] = {k: round(v, 4) for k, v in q_values.items()}

        return result

    async def _execute_debate(
        self,
        client: AsyncOpenAI,
        graph: Any,
        budget: APIBudget,
        event: TickElapsedEvent,
        result: dict[str, Any],
        stimulus_ctx: str,
    ) -> None:
        """Run a DEBATE action: call LLM, add argument, satiate drive."""
        debate_result = await self._call_llm_debate(
            client, graph, stimulus_context=stimulus_ctx
        )
        if debate_result and debate_result.claim:
            budget.record_call(self.agent_id, event.tick)
            node_id = await graph.add_argument_async(
                agent_id=self.agent_id,
                claim=debate_result.claim,
                target_node_id=debate_result.target_node_id,
                attack_type=debate_result.attack_type,
                tick=event.tick,
            )
            delta_phi = graph.calculate_phi_star_proxy(node_id)
            self.drive.satiate(delta_phi, alpha=2.0)
            self.memory.add_experience(Experience(
                state_summary=graph.get_recent_context(3),
                action="DEBATE",
                claim=debate_result.claim,
                delta_phi=delta_phi,
                tick=event.tick,
            ))
            result.update({
                "action_type": "DEBATE",
                "node_id": node_id,
                "claim": debate_result.claim,
                "target_node_id": debate_result.target_node_id,
                "attack_type": debate_result.attack_type,
                "delta_phi": delta_phi,
            })
        else:
            result["action_type"] = "PASS"

    # ──────────────────────────────────────────────────────────────────────────
    # Internal action executors
    # ──────────────────────────────────────────────────────────────────────────

    async def _do_search(self) -> None:
        """Retrieve a domain fact and store in semantic memory."""
        pair = get_search_result(self.memory)
        if pair:
            concept, fact = pair
            self.memory.add_fact(concept, fact)
            self.drive.satiate(0.02)
            logger.debug("%s SEARCH -> added fact: %s", self.agent_id, concept)

    async def _do_read(self, tick: int) -> None:
        """Absorb the most recent working-memory entry as a semantic fact."""
        if self.memory.working:
            entry = list(self.memory.working)[-1]
            self.memory.add_fact(f"observacion_t{tick}", entry)
            self.drive.satiate(0.02)
            logger.debug("%s READ -> stored working observation at tick %d", self.agent_id, tick)

    async def _call_llm_debate(
        self, client: AsyncOpenAI, graph: Any, stimulus_context: str = ""
    ) -> AgentAction | None:
        """Call the LLM asynchronously and return a parsed AgentAction."""
        graph_context = graph.get_recent_context(last_n=6)
        target_ids = graph.valid_target_ids()

        discourse_query = graph_context[:200] if graph_context else None
        memory_ctx = self.memory.get_prompt_context(discourse_query=discourse_query)

        system_msg = SYSTEM_PROMPT_TEMPLATE.format(role_prompt=self.role_prompt)
        user_msg = USER_PROMPT_TEMPLATE.format(
            graph_context=graph_context or "No hay argumentos aun. Presenta tu posicion inicial.",
            target_ids=", ".join(target_ids) if target_ids else "Ninguno -- crea un argumento raiz (target_node_id: null)",
            deficit=self.drive.deficit,
            memory_context=memory_ctx,
            stimulus_context=stimulus_context or "No specific stimulus. Act proactively.",
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                extra = {"seed": self._llm_seed} if self._llm_seed is not None else {}
                response = await client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.7,
                    max_tokens=300,
                    timeout=REQUEST_TIMEOUT,
                    **extra,
                )
                raw = response.choices[0].message.content or ""
                raw = self._extract_json(raw)
                action = AgentAction.model_validate_json(raw)
                logger.info(
                    "%s DEBATE (deficit=%.4f, attempt=%d)",
                    self.agent_id, self.drive.deficit, attempt + 1,
                )
                return action

            except (ValidationError, json.JSONDecodeError) as exc:
                logger.warning(
                    "%s parse error (attempt %d/%d): %s",
                    self.agent_id, attempt + 1, MAX_RETRIES + 1, exc,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.0)
                    continue
                self.drive.deficit += 0.1

            except Exception as exc:
                logger.warning(
                    "%s LLM error (attempt %d/%d): %s",
                    self.agent_id, attempt + 1, MAX_RETRIES + 1, type(exc).__name__,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                self.drive.deficit += 0.1

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Baseline sync path (LangGraph / round-robin modes)
    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self,
        graph_context: str,
        target_ids: list[str],
    ) -> AgentAction | None:
        """Synchronous forced-action path for baseline orchestration modes.

        Bypasses sigmoid sampling and utility selection -- the simulation decides
        when this agent speaks. Drive mechanics still apply for logging continuity.
        """
        self.drive.decay()

        system_msg = SYSTEM_PROMPT_TEMPLATE.format(role_prompt=self.role_prompt)
        user_msg = USER_PROMPT_TEMPLATE.format(
            graph_context=graph_context or "No hay argumentos aun. Presenta tu posicion inicial.",
            target_ids=", ".join(target_ids) if target_ids else "Ninguno -- crea un argumento raiz (target_node_id: null)",
            deficit=self.drive.deficit,
            memory_context=self.memory.get_prompt_context(),
            stimulus_context="Externally routed -- respond to current discourse state.",
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                extra = {"seed": self._llm_seed} if self._llm_seed is not None else {}
                response = self._sync_client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.7,
                    max_tokens=300,
                    timeout=REQUEST_TIMEOUT,
                    **extra,
                )
                raw = response.choices[0].message.content or ""
                raw = self._extract_json(raw)
                action = AgentAction.model_validate_json(raw)
                return action

            except (ValidationError, json.JSONDecodeError) as exc:
                logger.warning("%s baseline parse error (attempt %d): %s", self.agent_id, attempt + 1, exc)
                if attempt < MAX_RETRIES:
                    import time; time.sleep(1.0)
                    continue
                self.drive.deficit += 0.1
                return None

            except Exception as exc:
                logger.warning("%s baseline LLM error (attempt %d): %s", self.agent_id, attempt + 1, type(exc).__name__)
                if attempt < MAX_RETRIES:
                    import time; time.sleep(1.0 * (attempt + 1))
                    continue
                self.drive.deficit += 0.1
                return None

        return None

    def learn(self, state: str, action: AgentAction, delta_phi: float) -> None:
        """Update drive and memory after a DEBATE action (baseline path)."""
        self.drive.satiate(delta_phi)
        self.memory.add_experience(Experience(
            state_summary=state[:200],
            action=action.action,
            claim=action.claim,
            delta_phi=delta_phi,
            tick=0,
        ))

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """Best-effort extraction of a JSON object from LLM output."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start: end + 1]
        return text
