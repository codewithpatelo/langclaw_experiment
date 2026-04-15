"""LangClaw agent -- event-driven AOP entity with homeostatic regulation.

Architecture
============
Each agent is an autonomous event-processing entity with two states:

  ACTIVE  — sensing stimuli, buffering events, computing needs.
  WORKING — deliberating via the cognitive loop (THINK → PLAN → EXECUTE → OBSERVE).

While WORKING the agent does NOT process new events; they buffer in the
asyncio queue and are drained on re-entry to ACTIVE.

Cognitive loop (within WORKING)
-------------------------------
  TRIAGE  — drain all buffered events, evaluate expected utility of each
            via StimulusEvaluator, select the highest-utility event.
  THINK   — assess pressure (deficit + drive + context + messages).
            If insufficient pressure → PASS → return to ACTIVE.
  PLAN    — elaborate a task plan (ordered list of actions).
  EXECUTE — execute each task: DEBATE, SEARCH, READ, MESSAGE.
  OBSERVE — log results, compute reward, Q-update.
            If incomplete → loop back to THINK.
            If done → emit response → return to ACTIVE.

Communication
-------------
  NewArgumentEvent   — public discourse (broadcast to all agents).
  DirectMessageEvent — private agent-to-agent (FIPA ACL performatives).
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
from langclaw.events import (
    DirectMessageEvent,
    NewArgumentEvent,
    SimulationEndEvent,
    TickElapsedEvent,
)
from langclaw.homeostasis import EpistemicDrive
from langclaw.memory import AgentMemory, Experience
from langclaw.q_learner import HomeostaticQLearner
from langclaw.schemas import AgentAction, AgentState, CognitivePhase

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
- "action": one of "PASS", "DEBATE", "SEARCH", "READ", "MESSAGE"
- "claim": your argument text (required if action is DEBATE, null otherwise)
- "target_node_id": the ID of the node you attack (use an ID from the list below, or null for a new root claim)
- "attack_type": either "undercut" or "rebuttal" (required if action is DEBATE, null otherwise)
- "send_to": target agent ID (required if action is MESSAGE, null otherwise)
- "message_content": your message text (required if action is MESSAGE, null otherwise)
- "message_type": one of "request", "inform", "propose", "confirm", "query" (required if action is MESSAGE, null otherwise)

Example DEBATE:
{{"action": "DEBATE", "claim": "Las cifras oficiales contradicen el informe", "target_node_id": "GOV-S1_abc123", "attack_type": "rebuttal", "send_to": null, "message_content": null, "message_type": null}}

Example MESSAGE:
{{"action": "MESSAGE", "claim": null, "target_node_id": null, "attack_type": null, "send_to": "GOV-S1", "message_content": "Attack claim OPP-S1_xyz — it contradicts our data", "message_type": "request"}}

Example PASS:
{{"action": "PASS", "claim": null, "target_node_id": null, "attack_type": null, "send_to": null, "message_content": null, "message_type": null}}

CRITICAL: Output ONLY the JSON object. No markdown, no explanation, no code fences.
"""

USER_PROMPT_TEMPLATE = """\
## Current Debate State
{graph_context}

## Available target node IDs you can attack
{target_ids}

## Unanswered attacks on your faction
{undefended_attacks}

## Your faction agents (you can MESSAGE them)
{faction_agents}

## Your Internal State
- Epistemic deficit: {deficit:.4f} (higher = more urgency to act)
- Drive: {drive:.4f}

## Your Recent Experience and Knowledge
{memory_context}

## Incoming Messages from Faction
{messages_context}

## Stimulus Context
{stimulus_context}

Respond now with your JSON action:"""

LANGGRAPH_USER_PROMPT_TEMPLATE = """\
## Current Debate State
{graph_context}

## Available target node IDs you can attack
{target_ids}

## Unanswered attacks on your faction
{undefended_attacks}

## Your faction agents (you can MESSAGE them)
{faction_agents}

## Your Internal State
- Epistemic deficit: {deficit:.4f} (higher = more urgency to act)

## Your Recent Experience and Knowledge
{memory_context}

## Incoming Messages from Faction
{messages_context}

## Stimulus Context
{stimulus_context}

Respond now with your JSON action:"""

# --- Operational constants (functional constraints) ---
# MAX_RETRIES: 4 retries (5 attempts total). With 10 concurrent agents
# sharing a rate-limited API, transient 429s need more headroom than a
# single-agent scenario.  Exponential backoff (1s, 2s, 4s, 8s) ensures
# the burst clears within ~15s.
MAX_RETRIES = 4
# REQUEST_TIMEOUT: 60s accounts for gpt-5-mini reasoning overhead
# (typical generation 5–15s including reasoning tokens, with margin).
REQUEST_TIMEOUT = 60.0
# MAX_COGNITIVE_LOOPS: the cognitive loop can re-plan up to 3 times per
# tick if the Q-learner picks an action that fails.  3 is the minimum
# that covers the common failure modes (stimulus-debate → proactive-debate
# → fallback action).
MAX_COGNITIVE_LOOPS = 3
# Note: temperature is not configurable for gpt-5-nano (reasoning model,
# only default=1 is supported).  Reasoning effort is left at model default
# (medium) to preserve debate quality.
# LLM_MAX_TOKENS: ceiling for reasoning + output tokens.
# OpenAI reasoning guide: "reserve at least 25,000 tokens for
# reasoning and outputs."  Actual usage is ~1000 tokens (model
# self-regulates), so cost is independent of this ceiling.
LLM_MAX_TOKENS_DEBATE = 50000
LLM_MAX_TOKENS_ANALYSIS = 50000


class LangClawAgent:
    """Event-driven AOP agent with homeostatic regulation, cognitive loop,
    and FIPA ACL directed messaging.

    Parameters
    ----------
    agent_id        : unique identifier (e.g. "GOV-S1")
    role_prompt     : VSM persona injected into the system prompt
    vsm_system      : VSM subsystem number ("S1".."S5")
    faction_agents  : list of agent IDs in the same faction
    """

    def __init__(
        self,
        agent_id: str,
        role_prompt: str,
        base_url: str,
        api_key: str,
        model: str = "gpt-5-nano",
        rng_seed: int | None = None,
        llm_seed: int | None = None,
        initial_deficit: float = 0.5,
        vsm_system: str | None = None,
        faction_agents: list[str] | None = None,
        stimulus_weights: dict[str, float] | None = None,
        debate_alpha: float = 2.0,
    ) -> None:
        self.agent_id = agent_id
        self.role_prompt = role_prompt
        self.vsm_system = vsm_system
        self.faction_agents = faction_agents or []
        self._debate_alpha = debate_alpha
        self.drive = EpistemicDrive(initial_deficit=initial_deficit)
        self.memory = AgentMemory(agent_id=agent_id)
        self.stimulus_evaluator = StimulusEvaluator(**(stimulus_weights or {}))
        self.utility_selector = UtilitySelector()
        self.q_learner = HomeostaticQLearner(
            eta=0.01, gamma=0.95, epsilon=0.1, rng_seed=rng_seed,
        )
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._rng = random.Random(rng_seed)
        self._llm_seed = llm_seed
        self._sync_client = OpenAI(base_url=base_url, api_key=api_key)

        self._event_buffer: list[NewArgumentEvent] = []
        self._message_buffer: list[DirectMessageEvent] = []
        self.state = AgentState.ACTIVE
        self._pending_messages: list[dict[str, str]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # HRRL async path (event-driven with cognitive loop)
    # ──────────────────────────────────────────────────────────────────────────

    async def run(
        self,
        event_queue: asyncio.Queue,
        graph: Any,
        budget: APIBudget,
        output_queue: asyncio.Queue,
        llm_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """Main AOP coroutine with state machine."""
        self._llm_semaphore = llm_semaphore
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

                if isinstance(event, DirectMessageEvent):
                    if event.to_agent == self.agent_id:
                        self._message_buffer.append(event)
                    event_queue.task_done()
                    continue

                if isinstance(event, TickElapsedEvent):
                    self.state = AgentState.WORKING
                    result = await self._cognitive_loop(event, graph, budget, client)
                    self.state = AgentState.ACTIVE
                    await output_queue.put(result)
                    event_queue.task_done()

    async def _cognitive_loop(
        self,
        event: TickElapsedEvent,
        graph: Any,
        budget: APIBudget,
        client: AsyncOpenAI,
    ) -> dict[str, Any]:
        """Execute the TRIAGE → THINK → PLAN → EXECUTE → OBSERVE loop.

        This is the HRRL path: the agent's internal state (deficit → sigmoid)
        decides whether to act. The Q-learner selects actions. Reward is
        homeostatic drive reduction. This loop is endogenous.

        The LangGraph path uses a compiled StateGraph instead — same cognitive
        phases but with the graph's conditional edges controlling flow.
        """
        deficit_before = self.drive.deficit

        # 1. Basal decay
        self.drive.decay()
        self.memory.update_working_tick(event.tick)

        # 2. TRIAGE: drain all buffered events, evaluate, select best
        buffered_events = list(self._event_buffer)
        self._event_buffer.clear()
        buffered_messages = list(self._message_buffer)
        self._message_buffer.clear()
        self._pending_messages.clear()

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

        state_features = HomeostaticQLearner.build_features(
            deficit=self.drive.deficit,
            graph_density=graph_density,
            n_stimuli=len(buffered_events),
            n_messages=len(buffered_messages),
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
            "n_messages_received": len(buffered_messages),
            "reward": 0.0,
            "q_values": {k: round(v, 4) for k, v in q_values.items()},
            "agent_state": AgentState.WORKING.value,
            "cognitive_phase": CognitivePhase.TRIAGE.value,
            "messages": [],
            "vsm_system": self.vsm_system,
        }

        # 3. THINK: assess pressure
        if not budget.can_call(self.agent_id, self.drive.deficit, event.tick):
            result["cognitive_phase"] = CognitivePhase.THINK.value
            return result

        # Endogenous activation gate: sigmoid decides
        if self._rng.random() >= activation_prob:
            result["cognitive_phase"] = CognitivePhase.THINK.value
            return result

        # Select best stimulus
        best_stimulus_event: NewArgumentEvent | None = None
        best_stimulus_utility = 0.0
        if buffered_events:
            stimulus_scored = []
            for evt in buffered_events:
                u = self.stimulus_evaluator.evaluate(evt, self.agent_id, self.memory, graph)
                stimulus_scored.append((evt, u))
            best_stimulus_event, best_stimulus_utility = max(stimulus_scored, key=lambda x: x[1])

        # 4. PLAN + EXECUTE: Q-learner selects action, may loop
        for loop_iter in range(MAX_COGNITIVE_LOOPS):
            result["cognitive_phase"] = CognitivePhase.PLAN.value
            q_action = self.q_learner.select_action(state_features)

            result["cognitive_phase"] = CognitivePhase.EXECUTE.value

            if q_action == "DEBATE_STIMULUS" and best_stimulus_event:
                result["stimulus_event_id"] = best_stimulus_event.node_id
                result["stimulus_utility"] = best_stimulus_utility
                stimulus_ctx = (
                    f"You are responding to: [{best_stimulus_event.node_id}] "
                    f"{best_stimulus_event.agent_id} ({best_stimulus_event.faction}): "
                    f'"{best_stimulus_event.claim}"'
                )
                await self._execute_debate(client, graph, budget, event, result, stimulus_ctx, buffered_messages)
                break

            elif q_action in ("DEBATE_STIMULUS", "DEBATE_PROACTIVE"):
                await self._execute_debate(client, graph, budget, event, result, "", buffered_messages)
                break

            elif q_action == "MESSAGE":
                await self._execute_message(client, graph, event, result, buffered_messages)
                break

            elif q_action == "SEARCH":
                result["action_type"] = "SEARCH"
                await self._do_search()
                budget.record_call(self.agent_id, event.tick)
                break

            elif q_action == "READ":
                result["action_type"] = "READ"
                await self._do_read(client, graph, event.tick)
                budget.record_call(self.agent_id, event.tick)
                break

        # 5. OBSERVE: compute reward, TD update
        result["cognitive_phase"] = CognitivePhase.OBSERVE.value
        result["deficit_after"] = self.drive.deficit

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
            n_messages=0,
        )
        self.q_learner.update(state_features, q_action, reward, next_features)

        result["reward"] = round(reward, 6)
        result["q_values"] = {
            k: round(v, 4) for k, v in self.q_learner.get_q_values(next_features).items()
        }
        result["utility_scores"] = {k: round(v, 4) for k, v in q_values.items()}
        result["messages"] = list(self._pending_messages)
        self._pending_messages.clear()

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Action executors
    # ──────────────────────────────────────────────────────────────────────────

    async def _execute_debate(
        self,
        client: AsyncOpenAI,
        graph: Any,
        budget: APIBudget,
        event: TickElapsedEvent,
        result: dict[str, Any],
        stimulus_ctx: str,
        messages: list[DirectMessageEvent],
    ) -> None:
        """Run a DEBATE action: call LLM, add argument, satiate drive."""
        debate_result = await self._call_llm(
            client, graph, stimulus_context=stimulus_ctx, messages=messages
        )
        if debate_result and debate_result.action == "MESSAGE" and debate_result.send_to:
            self._handle_message_output(debate_result, result)
            return

        if debate_result and debate_result.action == "DEBATE" and debate_result.claim:
            budget.record_call(self.agent_id, event.tick)
            node_id = await graph.add_argument_async(
                agent_id=self.agent_id,
                claim=debate_result.claim,
                target_node_id=debate_result.target_node_id,
                attack_type=debate_result.attack_type,
                tick=event.tick,
            )
            delta_phi = graph.calculate_phi_star_proxy(node_id)
            self.drive.satiate(delta_phi, alpha=self._debate_alpha)
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

    async def _execute_message(
        self,
        client: AsyncOpenAI,
        graph: Any,
        event: TickElapsedEvent,
        result: dict[str, Any],
        messages: list[DirectMessageEvent],
    ) -> None:
        """Run a MESSAGE action: call LLM to compose a directed message."""
        msg_result = await self._call_llm(
            client, graph,
            stimulus_context="You should coordinate with your faction. Send a MESSAGE.",
            messages=messages,
        )
        if msg_result and msg_result.action == "MESSAGE" and msg_result.send_to:
            self._handle_message_output(msg_result, result)
        elif msg_result and msg_result.action == "DEBATE" and msg_result.claim:
            await self._execute_debate(
                client, graph, APIBudget(hard_limit=999), event, result, "", messages
            )
        else:
            result["action_type"] = "PASS"

    def _handle_message_output(self, action: AgentAction, result: dict[str, Any]) -> None:
        """Process a MESSAGE action from LLM output.

        Satiation is proportional to message substance: longer, directed
        messages with a valid recipient reduce deficit more.  An empty or
        malformed message provides zero satiation.
        """
        result["action_type"] = "MESSAGE"
        result["send_to"] = action.send_to
        result["message_content"] = action.message_content
        result["message_type"] = action.message_type
        self._pending_messages.append({
            "to_agent": action.send_to or "",
            "content": action.message_content or "",
            "msg_type": action.message_type or "inform",
        })
        content = (action.message_content or "").strip()
        has_recipient = action.send_to in self.faction_agents
        word_count = len(content.split()) if content else 0
        substance = min(1.0, word_count / 20.0) if has_recipient else 0.0
        self.drive.satiate(substance * 0.08, alpha=1.0)

    async def _do_search(self) -> None:
        """Retrieve a domain fact and store in semantic memory.

        Satiation is proportional to novelty: if the agent already has
        semantically similar facts, the marginal value is low.  Measured
        as 1 − (number of existing similar facts / threshold).
        """
        concept, fact = get_search_result(self.memory)
        existing = self.memory.search_relevant(fact, "semantic", limit=3)
        novelty = max(0.0, 1.0 - len(existing) / 3.0)
        self.memory.add_fact(concept, fact)
        self.drive.satiate(novelty * 0.06, alpha=1.0)
        logger.debug(
            "%s SEARCH -> %s (novelty=%.2f, satiation=%.4f)",
            self.agent_id, concept, novelty, novelty * 0.06,
        )

    async def _do_read(self, client: AsyncOpenAI, graph: Any, tick: int) -> None:
        """Synthesize the current argument graph into a strategic assessment.

        The agent reads the full discourse state, asks the LLM for a concise
        strategic analysis (weaknesses, opportunities, uncontested claims),
        and stores the result in semantic memory for future debates.
        """
        context = graph.get_recent_context(last_n=12)
        if not context or "No arguments" in context:
            logger.debug("%s READ -> no discourse to analyse at tick %d", self.agent_id, tick)
            return

        summary = graph.get_state_summary()
        prompt = (
            f"You are {self.agent_id}. Analyze the current debate state and produce "
            f"a BRIEF strategic assessment (max 150 words, in Spanish).\n\n"
            f"Graph: {summary['nodes']} nodes, {summary['edges']} edges\n\n"
            f"Recent arguments:\n{context}\n\n"
            f"Your role: {self.role_prompt[:200]}\n\n"
            f"Identify: 1) Weakest opponent arguments to attack, "
            f"2) Uncontested claims that need response, "
            f"3) Your faction's strongest line of argument.\n"
            f"Reply with ONLY the strategic assessment, no JSON."
        )

        try:
            sem = getattr(self, "_llm_semaphore", None)
            if sem is not None:
                await sem.acquire()
            try:
                response = await client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=LLM_MAX_TOKENS_ANALYSIS,
                    timeout=REQUEST_TIMEOUT,
                )
            finally:
                if sem is not None:
                    sem.release()
            analysis = (response.choices[0].message.content or "").strip()
            if analysis:
                self.memory.add_fact(f"strategic_read_t{tick}", analysis[:400])
                word_count = len(analysis.split())
                has_references = any(
                    tag in analysis for tag in ("GOV-", "OPP-", "S1", "S2", "S3", "S4", "S5")
                )
                depth = min(1.0, word_count / 80.0)
                specificity = 0.3 if has_references else 0.0
                quality = min(1.0, depth + specificity)
                self.drive.satiate(quality * 0.06, alpha=1.0)
                logger.debug(
                    "%s READ -> analysis at tick %d (quality=%.2f): %s...",
                    self.agent_id, tick, quality, analysis[:80],
                )
        except Exception as exc:
            logger.warning("%s READ LLM error at tick %d: %s", self.agent_id, tick, exc)

    async def _call_llm(
        self,
        client: AsyncOpenAI,
        graph: Any,
        stimulus_context: str = "",
        messages: list[DirectMessageEvent] | None = None,
    ) -> AgentAction | None:
        """Call the LLM and return a parsed AgentAction."""
        graph_context = graph.get_recent_context(last_n=6)
        target_ids = graph.valid_target_ids()

        discourse_query = graph_context[:200] if graph_context else None
        memory_ctx = self.memory.get_prompt_context(discourse_query=discourse_query)

        messages_ctx = "No messages."
        if messages:
            msg_lines = []
            for m in messages:
                msg_lines.append(
                    f"[{m.performative.upper()}] from {m.from_agent}: {m.content}"
                )
            messages_ctx = "\n".join(msg_lines)

        faction_agents_str = ", ".join(
            a for a in self.faction_agents if a != self.agent_id
        ) or "None"

        faction_prefix = self.agent_id.split("-")[0] + "-"
        undefended = graph.get_undefended_attacks(faction_prefix)
        undefended_ctx = "None — your faction's claims are all defended."
        if undefended:
            lines = []
            for u in undefended[:5]:
                lines.append(
                    f"- [{u['attacker_node']}] attacks your [{u['attacked_node']}]: "
                    f"\"{u['attacker_claim']}\""
                )
            undefended_ctx = "\n".join(lines)

        system_msg = SYSTEM_PROMPT_TEMPLATE.format(role_prompt=self.role_prompt)
        user_msg = USER_PROMPT_TEMPLATE.format(
            graph_context=graph_context or "No hay argumentos aun. Presenta tu posicion inicial.",
            target_ids=", ".join(target_ids) if target_ids else "Ninguno -- crea un argumento raiz (target_node_id: null)",
            faction_agents=faction_agents_str,
            deficit=self.drive.deficit,
            drive=self.drive.drive_value,
            undefended_attacks=undefended_ctx,
            memory_context=memory_ctx,
            messages_context=messages_ctx,
            stimulus_context=stimulus_context or "No specific stimulus. Act proactively.",
        )

        sem = getattr(self, "_llm_semaphore", None)

        for attempt in range(MAX_RETRIES + 1):
            try:
                extra = {"seed": self._llm_seed} if self._llm_seed is not None else {}
                if sem is not None:
                    await sem.acquire()
                try:
                    response = await client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        max_completion_tokens=LLM_MAX_TOKENS_DEBATE,
                        timeout=REQUEST_TIMEOUT,
                        **extra,
                    )
                finally:
                    if sem is not None:
                        sem.release()
                raw = response.choices[0].message.content or ""
                raw = self._extract_json(raw)
                action = AgentAction.model_validate_json(raw)
                logger.info(
                    "%s %s (deficit=%.4f, attempt=%d)",
                    self.agent_id, action.action, self.drive.deficit, attempt + 1,
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
                is_rate_limit = "RateLimit" in type(exc).__name__
                logger.warning(
                    "%s LLM error (attempt %d/%d): %s%s",
                    self.agent_id, attempt + 1, MAX_RETRIES + 1,
                    type(exc).__name__,
                    f" | {exc}" if attempt == 0 else "",
                )
                if attempt < MAX_RETRIES:
                    base = 5.0 if is_rate_limit else 2.0
                    backoff = base * (2.0 ** attempt) + self._rng.random() * 2
                    await asyncio.sleep(backoff)
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
        incoming_messages: list[DirectMessageEvent] | None = None,
        stimulus_context: str = "Externally routed -- respond to current discourse state.",
        undefended_attacks_ctx: str = "None — your faction's claims are all defended.",
    ) -> AgentAction | None:
        """Synchronous externally-triggered action path (LangGraph baseline).

        Same action space, memory, and cognitive capabilities as HRRL.
        Key differences:
          - Activation decided by the external router, not sigmoid/Q-learner.
          - Prompt shows epistemic deficit (observable state) but NOT drive
            D(δ)=(δ-ε)² — drive is HRRL's mathematical transformation.
          - No Q-learner selects the action; the LLM decides openly.

        Note: drive.decay() is NOT called here — the simulation loop handles
        deficit decay for all agents uniformly before invoking step().
        """

        messages_ctx = "No messages."
        if incoming_messages:
            msg_lines = [
                f"[{m.performative.upper()}] from {m.from_agent}: {m.content}"
                for m in incoming_messages
            ]
            messages_ctx = "\n".join(msg_lines)

        faction_agents_str = ", ".join(
            a for a in self.faction_agents if a != self.agent_id
        ) or "None"

        discourse_query = graph_context[:200] if graph_context else None
        memory_ctx = self.memory.get_prompt_context(discourse_query=discourse_query)

        system_msg = SYSTEM_PROMPT_TEMPLATE.format(role_prompt=self.role_prompt)
        user_msg = LANGGRAPH_USER_PROMPT_TEMPLATE.format(
            graph_context=graph_context or "No hay argumentos aun. Presenta tu posicion inicial.",
            target_ids=", ".join(target_ids) if target_ids else "Ninguno -- crea un argumento raiz (target_node_id: null)",
            undefended_attacks=undefended_attacks_ctx,
            faction_agents=faction_agents_str,
            deficit=self.drive.deficit,
            memory_context=memory_ctx,
            messages_context=messages_ctx,
            stimulus_context=stimulus_context,
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
                    max_completion_tokens=LLM_MAX_TOKENS_DEBATE,
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
        self.drive.satiate(delta_phi, alpha=self._debate_alpha)
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
