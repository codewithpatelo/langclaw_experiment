"""AgentMemory: three-layer memory backed by LangGraph's InMemoryStore.

  Episodic  -- past (state, action, reward) experiences, retrievable by
               semantic similarity to the current discourse context.
  Semantic  -- domain facts injected by SEARCH actions; semantic retrieval.
  Working   -- sliding window of recently observed arguments (graph context).

Each layer is a namespace inside a shared InMemoryStore instance, enabling
semantic search across all stored items.
"""

from __future__ import annotations

import os
import uuid
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph.store.memory import InMemoryStore

if TYPE_CHECKING:
    from langclaw.events import NewArgumentEvent

logger = logging.getLogger(__name__)


def _build_store() -> InMemoryStore:
    """Create InMemoryStore with optional semantic search via embeddings."""
    try:
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key,
            )
            return InMemoryStore(
                index={"embed": embeddings, "dims": 1536}
            )
    except Exception as exc:
        logger.warning("Embeddings unavailable (%s); falling back to non-indexed store", exc)

    return InMemoryStore()


# Module-level shared store: all agents share one InMemoryStore,
# isolated by agent_id namespaces.
_shared_store: InMemoryStore | None = None


def get_shared_store() -> InMemoryStore:
    global _shared_store
    if _shared_store is None:
        _shared_store = _build_store()
    return _shared_store


def reset_shared_store() -> None:
    """Reset the shared store between benchmark runs."""
    global _shared_store
    _shared_store = None


@dataclass
class Experience:
    state_summary: str
    action: str          # "DEBATE" | "SEARCH" | "READ" | "PASS"
    claim: str | None
    delta_phi: float
    tick: int


class AgentMemory:
    """Three-layer memory backed by LangGraph InMemoryStore.

    Namespaces:
      (agent_id, "episodic")  -- experiences with delta_phi, action, claim
      (agent_id, "semantic")  -- facts from SEARCH
      (agent_id, "working")   -- recent observations from other agents
    """

    MAX_EPISODIC = 20
    MAX_WORKING = 10
    PROMPT_EPISODIC = 3
    PROMPT_SEMANTIC = 3

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._store = get_shared_store()
        self._has_index = hasattr(self._store, '_index') or True
        # Local working memory deque for fast access (also mirrored to store)
        self.working: deque[str] = deque(maxlen=self.MAX_WORKING)
        # Local episodic cache for backward compat with utility functions
        self._episodic_cache: list[Experience] = []
        # Local semantic cache for backward compat
        self._semantic_cache: dict[str, str] = {}

    def _ns(self, layer: str) -> tuple[str, str]:
        return (self.agent_id, layer)

    def add_experience(self, experience: Experience) -> None:
        """Store a new experience in episodic memory."""
        item_id = str(uuid.uuid4())[:8]
        self._store.put(
            self._ns("episodic"),
            item_id,
            {
                "text": f"Action={experience.action}, Claim={experience.claim or 'N/A'}, "
                        f"Reward(dphi)={experience.delta_phi:.4f}, Tick={experience.tick}",
                "action": experience.action,
                "claim": experience.claim,
                "delta_phi": experience.delta_phi,
                "tick": experience.tick,
                "state_summary": experience.state_summary[:200],
            },
        )
        self._episodic_cache.append(experience)
        self._episodic_cache = sorted(
            self._episodic_cache,
            key=lambda e: abs(e.delta_phi),
            reverse=True,
        )[:self.MAX_EPISODIC]

    def add_fact(self, concept: str, fact: str) -> None:
        """Add a semantic fact (result of a SEARCH action)."""
        self._store.put(
            self._ns("semantic"),
            concept,
            {"text": fact, "concept": concept},
        )
        self._semantic_cache[concept] = fact

    def observe(self, event: "NewArgumentEvent") -> None:
        """Record an argument made by another agent into working memory."""
        entry = f"[{event.node_id}] {event.agent_id}: \"{event.claim}\""
        if event.target_node_id:
            entry += f" --[{event.attack_type}]--> {event.target_node_id}"
        self.working.append(entry)

        item_id = f"obs_{event.tick}_{event.node_id}"
        self._store.put(
            self._ns("working"),
            item_id,
            {
                "text": entry,
                "tick": event.tick,
                "source_agent": event.agent_id,
                "node_id": event.node_id,
            },
        )

    def update_working_tick(self, tick: int) -> None:
        """Called on each TickElapsedEvent; currently a no-op hook."""
        pass

    def search_relevant(self, query: str, layer: str = "semantic", limit: int = 3) -> list[dict[str, Any]]:
        """Semantic search over a memory layer. Returns list of item values."""
        try:
            results = self._store.search(
                self._ns(layer),
                query=query,
                limit=limit,
            )
            return [r.value for r in results]
        except Exception:
            return []

    def get_prompt_context(self, discourse_query: str | None = None) -> str:
        """Return compact prompt context combining episodic and semantic memories.

        If discourse_query is provided and the store supports semantic search,
        retrieves contextually relevant memories instead of a fixed slice.
        """
        parts: list[str] = []

        if discourse_query:
            episodic_results = self.search_relevant(discourse_query, "episodic", self.PROMPT_EPISODIC)
            if episodic_results:
                parts.append("=== Experiencias pasadas relevantes ===")
                for item in episodic_results:
                    parts.append(item.get("text", ""))
            else:
                parts.append("Sin experiencias previas.")

            semantic_results = self.search_relevant(discourse_query, "semantic", self.PROMPT_SEMANTIC)
            if semantic_results:
                parts.append("=== Conocimiento semantico relevante ===")
                for item in semantic_results:
                    parts.append(f"- {item.get('concept', '?')}: {item.get('text', '')}")
        else:
            if self._episodic_cache:
                parts.append("=== Experiencias pasadas (mejores/peores) ===")
                for exp in self._episodic_cache[:self.PROMPT_EPISODIC]:
                    parts.append(
                        f"Accion={exp.action}, Claim={exp.claim or 'N/A'}, "
                        f"Reward(dphi)={exp.delta_phi:.4f}"
                    )
            else:
                parts.append("Sin experiencias previas.")

            if self._semantic_cache:
                parts.append("=== Conocimiento semantico ===")
                for i, (concept, fact) in enumerate(self._semantic_cache.items()):
                    if i >= self.PROMPT_SEMANTIC:
                        break
                    parts.append(f"- {concept}: {fact}")

        return "\n".join(parts)

    def recent_avg_delta_phi(self, n: int = 5) -> float:
        """Average delta_phi of the last n DEBATE experiences."""
        debate_exps = [e for e in self._episodic_cache if e.action == "DEBATE"][:n]
        if not debate_exps:
            return 0.0
        return sum(e.delta_phi for e in debate_exps) / len(debate_exps)

    def semantic_density(self) -> float:
        """Fraction of semantic slots filled (0-1), capped at 1."""
        return min(1.0, len(self._semantic_cache) / max(1, self.PROMPT_SEMANTIC * 3))

    @property
    def episodic(self) -> list[Experience]:
        return self._episodic_cache

    @property
    def semantic(self) -> dict[str, str]:
        return self._semantic_cache

    def __len__(self) -> int:
        return len(self._episodic_cache)
