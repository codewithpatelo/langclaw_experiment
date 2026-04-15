"""
AgentMemory: three-layer memory system for LangClaw agents.

  Episodic  — past (state, action, reward) experiences, prioritised by |Δφ*|
              so the most and least informative contributions surface first.
  Semantic  — domain facts injected by SEARCH actions; grows over time.
  Working   — a sliding window of recently observed arguments (graph context).

Replaces the flat episodic_memory list in the original agent.py and enables
SEARCH-driven knowledge accumulation.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langclaw.events import NewArgumentEvent
    from langclaw.schemas import AgentAction


@dataclass
class Experience:
    state_summary: str
    action: str          # "DEBATE" | "SEARCH" | "READ" | "PASS"
    claim: str | None
    delta_phi: float
    tick: int


class AgentMemory:
    """
    Three-layer memory system.

    Episodic is capped at MAX_EPISODIC entries and sorted by |Δφ*|
    (highest absolute value first) so both successes and instructive
    failures stay visible to the LLM.
    """

    MAX_EPISODIC = 20
    MAX_WORKING = 10
    PROMPT_EPISODIC = 3   # how many episodic entries to inject per prompt
    PROMPT_SEMANTIC = 3   # how many semantic facts to inject per prompt

    def __init__(self) -> None:
        self.episodic: list[Experience] = []
        self.semantic: dict[str, str] = {}          # concept → fact
        self.working: deque[str] = deque(maxlen=self.MAX_WORKING)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_experience(self, experience: Experience) -> None:
        """Store a new experience, keeping only the most informative MAX_EPISODIC."""
        self.episodic.append(experience)
        self.episodic = sorted(
            self.episodic,
            key=lambda e: abs(e.delta_phi),
            reverse=True,
        )[: self.MAX_EPISODIC]

    def add_fact(self, concept: str, fact: str) -> None:
        """Add a semantic fact (result of a SEARCH action)."""
        self.semantic[concept] = fact

    def observe(self, event: "NewArgumentEvent") -> None:
        """Record an argument made by another agent into working memory."""
        entry = f"[{event.node_id}] {event.agent_id}: \"{event.claim}\""
        if event.target_node_id:
            entry += f" --[{event.attack_type}]--> {event.target_node_id}"
        self.working.append(entry)

    def update_working_tick(self, tick: int) -> None:
        """Called on each TickElapsedEvent; currently a no-op hook for future use."""
        pass

    # ------------------------------------------------------------------
    # Read / prompt context
    # ------------------------------------------------------------------

    def get_prompt_context(self) -> str:
        """
        Return a compact string combining:
          - Top PROMPT_EPISODIC episodic experiences (best/worst by |Δφ*|)
          - Top PROMPT_SEMANTIC semantic facts
        Injected into the LLM user prompt for in-context RL.
        """
        parts: list[str] = []

        if self.episodic:
            parts.append("=== Experiencias pasadas (mejores/peores) ===")
            for exp in self.episodic[: self.PROMPT_EPISODIC]:
                parts.append(
                    f"Acción={exp.action}, Claim={exp.claim or 'N/A'}, "
                    f"Reward(Δφ*)={exp.delta_phi:.4f}"
                )
        else:
            parts.append("Sin experiencias previas.")

        if self.semantic:
            parts.append("=== Conocimiento semántico ===")
            for i, (concept, fact) in enumerate(self.semantic.items()):
                if i >= self.PROMPT_SEMANTIC:
                    break
                parts.append(f"- {concept}: {fact}")

        return "\n".join(parts)

    def recent_avg_delta_phi(self, n: int = 5) -> float:
        """Average Δφ* of the last n DEBATE experiences. 0.0 if none."""
        debate_exps = [e for e in self.episodic if e.action == "DEBATE"][:n]
        if not debate_exps:
            return 0.0
        return sum(e.delta_phi for e in debate_exps) / len(debate_exps)

    def semantic_density(self) -> float:
        """Fraction of semantic slots filled (0–1), capped at 1."""
        return min(1.0, len(self.semantic) / max(1, self.PROMPT_SEMANTIC * 3))

    def __len__(self) -> int:
        return len(self.episodic)
