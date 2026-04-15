"""LangClaw agent — HRRL-regulated, event-driven, with utility-based action selection.

Two operating modes:
  HRRL (async) — agent runs as an asyncio coroutine, reacts to events from a queue,
                 selects actions via UtilitySelector, regulated by EpistemicDrive.
  Baseline (sync) — simulation forces the agent to call the LLM directly (round-robin
                    / random modes); drive mechanics bypassed.

HRRL cycle (async path):
    TickElapsedEvent → decay → sigmoid sample → utility select → execute action
    NewArgumentEvent → observe → update working memory (no forced action)
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from langclaw.actions import UtilitySelector, get_search_result
from langclaw.budget import APIBudget
from langclaw.events import NewArgumentEvent, SimulationEndEvent, TickElapsedEvent
from langclaw.homeostasis import EpistemicDrive
from langclaw.memory import AgentMemory, Experience
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

Respond now with your JSON action:"""

MAX_RETRIES = 2
REQUEST_TIMEOUT = 30.0


class LangClawAgent:
    """HRRL-regulated agent with utility-based action selection and rich memory.

    Parameters
    ----------
    agent_id       : unique identifier (e.g. "GOV-1")
    role_prompt    : political persona injected into the system prompt
    base_url       : OpenAI-compatible API endpoint
    api_key        : API key for the endpoint
    model          : model name for completions
    seed           : random seed for reproducibility
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
        self.memory = AgentMemory()
        self.utility_selector = UtilitySelector()
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._rng = random.Random(rng_seed)   # sigmoid sampling
        self._llm_seed = llm_seed             # forwarded to LLM API
        # Sync client kept for baseline step() compatibility
        self._sync_client = OpenAI(base_url=base_url, api_key=api_key)

    # ──────────────────────────────────────────────────────────────────────────
    # HRRL async path
    # ──────────────────────────────────────────────────────────────────────────

    async def run(
        self,
        event_queue: asyncio.Queue,
        graph: Any,
        budget: APIBudget,
        output_queue: asyncio.Queue,
    ) -> None:
        """Main HRRL coroutine. Processes events and puts tick results into output_queue.

        The simulation reads output_queue to handle graph updates, log entries,
        and event broadcasting — keeping agents decoupled from each other.
        """
        async with AsyncOpenAI(base_url=self._base_url, api_key=self._api_key) as client:
            while True:
                event = await event_queue.get()

                # ── Shutdown ──────────────────────────────────────────────────
                if isinstance(event, SimulationEndEvent):
                    event_queue.task_done()
                    break

                # ── Observe other agents' arguments (no action) ───────────────
                if isinstance(event, NewArgumentEvent):
                    if event.agent_id != self.agent_id:
                        self.memory.observe(event)
                    event_queue.task_done()
                    continue

                # ── Tick: decay + evaluate + maybe act ────────────────────────
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
        """Execute one HRRL tick and return a result dict for the simulation."""
        deficit_before = self.drive.deficit
        self.drive.decay()
        activation_prob = self.drive.get_activation_probability()
        self.memory.update_working_tick(event.tick)

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
        }

        # Rate limit check
        if not budget.can_call(self.agent_id, self.drive.deficit, event.tick):
            return result

        # Sigmoid sampling
        if self._rng.random() >= activation_prob:
            return result

        # Utility-based action selection
        graph_node_count = graph.get_state_summary()["nodes"]
        utility_scores = self.utility_selector.scores(self.drive, self.memory, graph_node_count)
        action_type = self.utility_selector.select(self.drive, self.memory, graph_node_count)
        result["utility_scores"] = {k: round(v, 4) for k, v in utility_scores.items()}
        result["action_type"] = action_type

        if action_type == "SEARCH":
            await self._do_search()
            budget.record_call(self.agent_id, event.tick)

        elif action_type == "READ":
            await self._do_read(event.tick)
            budget.record_call(self.agent_id, event.tick)

        elif action_type == "DEBATE":
            debate_result = await self._call_llm_debate(client, graph)
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
                # α=2.0: a good debate (Δφ*≈0.15) removes 0.30 deficit,
                # enough to cross back below θ=0.7 and create genuine oscillation
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

        result["deficit_after"] = self.drive.deficit
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Internal action executors
    # ──────────────────────────────────────────────────────────────────────────

    async def _do_search(self) -> None:
        """Retrieve a domain fact and store in semantic memory."""
        pair = get_search_result(self.memory)
        if pair:
            concept, fact = pair
            self.memory.add_fact(concept, fact)
            # Minimal satiation: gathering facts doesn't satisfy the drive to argue.
            # Only DEBATE (with α=2.0) produces meaningful deficit reduction.
            self.drive.satiate(0.02)
            logger.debug("%s SEARCH → added fact: %s", self.agent_id, concept)

    async def _do_read(self, tick: int) -> None:
        """Absorb the most recent working-memory entry as a semantic fact."""
        if self.memory.working:
            entry = list(self.memory.working)[-1]
            self.memory.add_fact(f"observacion_t{tick}", entry)
            self.drive.satiate(0.02)
            logger.debug("%s READ → stored working observation at tick %d", self.agent_id, tick)

    async def _call_llm_debate(
        self, client: AsyncOpenAI, graph: Any
    ) -> AgentAction | None:
        """Call the LLM asynchronously and return a parsed AgentAction."""
        graph_context = graph.get_recent_context(last_n=6)
        target_ids = graph.valid_target_ids()

        system_msg = SYSTEM_PROMPT_TEMPLATE.format(role_prompt=self.role_prompt)
        user_msg = USER_PROMPT_TEMPLATE.format(
            graph_context=graph_context or "No hay argumentos aún. Presenta tu posición inicial.",
            target_ids=", ".join(target_ids) if target_ids else "Ninguno — crea un argumento raíz (target_node_id: null)",
            deficit=self.drive.deficit,
            memory_context=self.memory.get_prompt_context(),
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
    # Baseline sync path (round-robin / random modes)
    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self,
        graph_context: str,
        target_ids: list[str],
    ) -> AgentAction | None:
        """Synchronous forced-action path for baseline orchestration modes.

        Bypasses sigmoid sampling and utility selection — the simulation decides
        when this agent speaks. Drive mechanics still apply for logging continuity.
        """
        self.drive.decay()
        activation_prob = self.drive.get_activation_probability()

        system_msg = SYSTEM_PROMPT_TEMPLATE.format(role_prompt=self.role_prompt)
        user_msg = USER_PROMPT_TEMPLATE.format(
            graph_context=graph_context or "No hay argumentos aún. Presenta tu posición inicial.",
            target_ids=", ".join(target_ids) if target_ids else "Ninguno — crea un argumento raíz (target_node_id: null)",
            deficit=self.drive.deficit,
            memory_context=self.memory.get_prompt_context(),
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                import time
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
