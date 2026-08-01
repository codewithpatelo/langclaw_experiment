"""LangGraph-style state-based LLM router for debate orchestration.

The LangGraphRouter implements the exogenous baseline: an external LLM reads
the current discourse state (last N arguments) and selects the agent whose
contribution would best advance the debate.

Key design decisions:
- The router prompt is NEUTRAL — it does not mention or optimise toward any
  of the evaluation metrics (AAF defeat cycles, PRR, Δφ*). This prevents
  the baseline from being biased toward the outcomes we are measuring.
- The router uses a dedicated LLM seed (via SeedFactory) for reproducibility.
- If the LLM call fails or returns an unrecognised agent ID, the router
  falls back to round-robin to preserve budget matching.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# Neutral router prompt — does NOT reference evaluation metrics.
# Includes the same structural features EPR agents use internally (deficit,
# stimulus_utility, recent_arguments, vsm_role) so the only difference between
# EPR and LangGraph is the locus of control (endogenous sigmoid vs exogenous router).
_ROUTER_PROMPT = """\
You are a neutral debate moderator with access to the internal state of the participants.

Recent debate arguments (most recent last):
{context}

Per-participant structural state (JSON, computed from the discourse graph):
{features_json}

Field meanings:
- deficit: epistemic deficit (higher = more accumulated tension to discharge)
- stimulus_utility: structural utility of responding to the most recent
  unanswered argument, computed via a multi-criteria sensor (faction relevance,
  target centrality, memory match, novelty, unanswered pressure)
- recent_arguments: number of arguments produced by this agent in the last 10 turns
- vsm_role: subsystem role (S1 Operations, S2 Coordination, S3 Control,
  S4 Intelligence, S5 Strategy)

Available participants: {agent_ids}

Select the participant whose contribution would best advance the discussion
at this point, taking BOTH the textual context AND the structural state into
account. Output ONLY the participant ID, exactly as listed above. No
explanation, no punctuation.\
"""


class LangGraphRouter:
    """State-based LLM router: reads discourse state, selects next agent.

    Analogous to a LangGraph conditional edge that routes based on the current
    state of the graph. The routing decision is made by an LLM, not by
    internal agent state — this is the key structural difference from HRRL.

    Parameters
    ----------
    base_url:
        OpenAI-compatible API base URL.
    api_key:
        API key for the LLM provider.
    model:
        Model ID used for routing decisions.
    seed:
        LLM seed for reproducibility (from SeedFactory.get("router_llm")).
    fallback_cycle:
        Whether to cycle through agents in order when the LLM call fails.
        Default True — ensures budget matching even under API failures.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        seed: int | None = None,
        fallback_cycle: bool = True,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._seed = seed
        self._fallback_cycle = fallback_cycle
        self._cycle_index: int = 0
        self._router_call_count: int = 0

    @property
    def router_call_count(self) -> int:
        """Total number of LLM calls made by the router (for cost transparency)."""
        return self._router_call_count

    def select_next_agent(
        self,
        discourse_context: str,
        agent_ids: list[str],
        agent_features: list[dict[str, Any]] | None = None,
        last_speaker: str | None = None,
    ) -> str:
        """Select the next agent to speak based on discourse state.

        Parameters
        ----------
        discourse_context:
            Textual summary of recent arguments (from ArgumentGraph.get_recent_context).
        agent_ids:
            All agent IDs eligible to speak this tick.
        agent_features:
            Per-agent structural features (deficit, stimulus_utility,
            recent_arguments, vsm_role).  When provided, the router sees the
            same internal state that EPR agents use for self-activation.
        last_speaker:
            The agent that spoke last (passed for context, not enforced).

        Returns
        -------
        Selected agent ID (guaranteed to be in agent_ids).
        """
        if not agent_ids:
            raise ValueError("agent_ids must be non-empty")

        features_json = (
            json.dumps(agent_features, ensure_ascii=False, indent=2)
            if agent_features
            else "[]"
        )
        prompt = _ROUTER_PROMPT.format(
            context=discourse_context or "No arguments have been made yet.",
            features_json=features_json,
            agent_ids=", ".join(agent_ids),
        )

        try:
            extra: dict = {"temperature": 0.0}
            if self._seed is not None:
                extra["seed"] = self._seed

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=50000,
                **extra,
            )
            self._router_call_count += 1

            raw = (response.choices[0].message.content or "").strip()
            # Accept exact match or case-insensitive prefix match
            for aid in agent_ids:
                if raw.upper() == aid.upper() or raw.upper().startswith(aid.upper()):
                    return aid

            logger.warning("Router returned unrecognised agent '%s'; falling back.", raw)

        except Exception as exc:
            logger.warning("Router LLM call failed (%s); falling back.", exc)

        return self._fallback(agent_ids)

    def _fallback(self, agent_ids: list[str]) -> str:
        """Round-robin fallback when the LLM call fails or returns invalid output."""
        if not self._fallback_cycle:
            return agent_ids[0]
        chosen = agent_ids[self._cycle_index % len(agent_ids)]
        self._cycle_index += 1
        return chosen

    def to_checkpoint(self) -> dict[str, object]:
        """Serialize router state for benchmark resume."""
        return {
            "cycle_index": self._cycle_index,
            "router_call_count": self._router_call_count,
        }

    def load_checkpoint(self, payload: dict[str, object]) -> None:
        """Restore router state from checkpoint."""
        self._cycle_index = int(payload.get("cycle_index", self._cycle_index))
        self._router_call_count = int(
            payload.get("router_call_count", self._router_call_count)
        )
