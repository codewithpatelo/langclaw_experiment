"""LangGraph router with access to structural features per agent.

This is the "fair baseline" introduced to address the information asymmetry
critique (REV1, REV2, REV3, REV5): the standard `LangGraphRouter` only sees
plain text, while HRRL agents see five structural features through the
StimulusEvaluator. To isolate "endogenous vs exogenous" from "informed vs
uninformed", this informed variant exposes the same structural state to the
exogenous router by augmenting the prompt with a JSON snapshot per agent:

    [
      {"agent_id": "GOV-S1", "deficit": 0.84, "stimulus_utility": 0.62,
       "recent_arguments": 3, "vsm_role": "S1"},
      ...
    ]

Computation reuses the existing `StimulusEvaluator` so the router sees the
same sensor signal that HRRL uses internally. The router's textual context
is preserved (last 6 arguments).

Falls back to the same round-robin behaviour as the parent router if the LLM
returns an invalid agent ID.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langclaw.router import LangGraphRouter

logger = logging.getLogger(__name__)


_INFORMED_ROUTER_PROMPT = """\
You are a neutral debate moderator with access to internal state of the participants.

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


class LangGraphInformedRouter(LangGraphRouter):
    """Router that receives structural features per agent in addition to text."""

    def select_next_agent_informed(
        self,
        discourse_context: str,
        agent_features: list[dict[str, Any]],
    ) -> str:
        """Select the next agent given textual context AND structural features.

        Parameters
        ----------
        discourse_context:
            Same textual summary used by the base router (last N arguments).
        agent_features:
            One dict per agent with at minimum {agent_id, deficit,
            stimulus_utility, recent_arguments, vsm_role}.

        Returns
        -------
        Selected agent ID, guaranteed to be in agent_features[*]['agent_id'].
        """
        if not agent_features:
            raise ValueError("agent_features must be non-empty")

        agent_ids = [f["agent_id"] for f in agent_features]
        features_json = json.dumps(agent_features, ensure_ascii=False, indent=2)

        prompt = _INFORMED_ROUTER_PROMPT.format(
            context=discourse_context or "No arguments have been made yet.",
            features_json=features_json,
            agent_ids=", ".join(agent_ids),
        )

        try:
            extra: dict = {}
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
            for aid in agent_ids:
                if raw.upper() == aid.upper() or raw.upper().startswith(aid.upper()):
                    return aid

            logger.warning(
                "InformedRouter returned unrecognised agent '%s'; falling back.", raw
            )

        except Exception as exc:
            logger.warning("InformedRouter LLM call failed (%s); falling back.", exc)

        return self._fallback(agent_ids)
