"""Action selection and stimulus evaluation for LangClaw agents.

Two systems coexist:
  UtilitySelector  -- legacy action-type scorer (used by baseline / fallback).
  StimulusEvaluator -- per-event multi-criteria utility (AOP redesign).

SEARCH provider
───────────────
Uses the Tavily Search API for real web search.  When TAVILY_API_KEY is not
set, falls back to a small static knowledge pool so the simulation still
runs offline.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, TYPE_CHECKING, Any

from langclaw.homeostasis import EpistemicDrive
from langclaw.memory import AgentMemory

if TYPE_CHECKING:
    from langclaw.events import NewArgumentEvent

logger = logging.getLogger(__name__)

ActionType = Literal["DEBATE", "SEARCH", "READ", "PASS"]

# ──────────────────────────────────────────────────────────────────────────────
# Fallback knowledge pool (used when TAVILY_API_KEY is not set)
# ──────────────────────────────────────────────────────────────────────────────
_FALLBACK_POOL: list[tuple[str, str]] = [
    ("PIB_caida", "El PIB cayó un 4.2% en el último trimestre según datos del banco central."),
    ("desempleo", "La tasa de desempleo alcanzó el 18.7%, la más alta en 15 años."),
    ("inflacion", "La inflación acumulada del año supera el 120%, erosionando el salario real."),
    ("deuda_publica", "La deuda pública representa el 89% del PIB, con vencimientos críticos en 6 meses."),
    ("reservas", "Las reservas internacionales cayeron a USD 2.100M, mínimo desde 2005."),
    ("gasto_social", "El gasto social fue recortado un 23% en términos reales durante la gestión actual."),
    ("corrupcion", "El índice de percepción de corrupción cayó 12 puntos en el último informe de Transparencia Internacional."),
    ("pobreza", "El índice de pobreza aumentó del 28% al 41% en los últimos tres años."),
    ("salud", "El 34% de los hospitales públicos reportan desabastecimiento de medicamentos esenciales."),
    ("emigracion", "Se estima que 480.000 ciudadanos emigraron en el último año, la cifra más alta registrada."),
]

# Queries the agent cycles through when calling Tavily
_SEARCH_QUERIES: list[str] = [
    "indicadores económicos gobierno crisis datos oficiales",
    "pobreza desempleo inflación estadísticas recientes",
    "corrupción gobierno contratos estatales investigaciones",
    "protestas sociales crisis política últimos meses",
    "deuda pública reservas internacionales riesgo país",
    "gasto social recortes presupuesto educación salud",
    "exportaciones competitividad tipo de cambio comercio",
    "emigración fuga de cerebros crisis migratoria",
    "independencia judicial poder judicial designaciones",
    "libertad de prensa medios concentración mediática",
]


class UtilitySelector:
    """
    Computes a utility score for each action and returns the argmax.

    Parameters
    ──────────
    theta : float
        Activation threshold (must match EpistemicDrive.theta). Default 0.7.
    k : float
        Sigmoid steepness. Default 10.
    debate_quality_floor : float
        If recent avg Δφ* falls below this, SEARCH becomes more attractive than DEBATE.
    semantic_density_floor : float
        If semantic density falls below this, READ is boosted.
    """

    def __init__(
        self,
        theta: float = 0.7,
        k: float = 10.0,
        debate_quality_floor: float = 0.15,
        semantic_density_floor: float = 0.2,
    ) -> None:
        self.theta = theta
        self.k = k
        self.debate_quality_floor = debate_quality_floor
        self.semantic_density_floor = semantic_density_floor

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def select(
        self,
        drive: EpistemicDrive,
        memory: AgentMemory,
        graph_node_count: int,
    ) -> ActionType:
        scores = self.scores(drive, memory, graph_node_count)
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def scores(
        self,
        drive: EpistemicDrive,
        memory: AgentMemory,
        graph_node_count: int,
    ) -> dict[ActionType, float]:
        return {
            "DEBATE": self._u_debate(drive, memory, graph_node_count),
            "SEARCH": self._u_search(drive, memory),
            "READ":   self._u_read(drive, memory),
            "PASS":   self._u_pass(drive),
        }

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    def _u_debate(
        self,
        drive: EpistemicDrive,
        memory: AgentMemory,
        graph_node_count: int,
    ) -> float:
        """
        High when:
          - δ is above threshold (drive is active)
          - graph has targets to attack
          - recent argument quality was decent
        """
        p_act = drive.get_activation_probability(k=self.k, theta=self.theta)
        has_targets = 1.0 if graph_node_count > 0 else 0.3
        avg_quality = memory.recent_avg_delta_phi()
        quality_factor = max(0.1, avg_quality / 0.5)   # normalised around 0.5 Δφ*
        return p_act * has_targets * quality_factor

    def _u_search(self, drive: EpistemicDrive, memory: AgentMemory) -> float:
        """
        High when:
          - δ is above threshold (drive active — agent needs to do something)
          - recent argument quality was poor (agent realises it needs better info)
        Low quality pushes toward search rather than repeating bad debates.
        """
        p_act = drive.get_activation_probability(k=self.k, theta=self.theta)
        avg_quality = memory.recent_avg_delta_phi()
        # quality deficit: 1.0 when avg_quality=0, 0.0 when avg_quality=0.5+
        quality_deficit = max(0.0, 1.0 - avg_quality / self.debate_quality_floor)
        # diminish if semantic memory is already full
        novelty = 1.0 - memory.semantic_density()
        return p_act * quality_deficit * novelty

    def _u_read(self, drive: EpistemicDrive, memory: AgentMemory) -> float:
        """
        High when:
          - δ is above threshold
          - semantic memory is sparse (agent lacks background knowledge)
        """
        p_act = drive.get_activation_probability(k=self.k, theta=self.theta)
        sparsity = max(0.0, self.semantic_density_floor - memory.semantic_density())
        return p_act * (sparsity / self.semantic_density_floor)

    def _u_pass(self, drive: EpistemicDrive) -> float:
        """
        Resting utility: high when the agent is sated (δ << θ).
        1 − sigmoid(δ) so that a fully sated agent (δ → 0) has U(PASS) ≈ 1.
        """
        return 1.0 - drive.get_activation_probability(k=self.k, theta=self.theta)


# ──────────────────────────────────────────────────────────────────────────────
# Stimulus evaluator (AOP redesign)
# ──────────────────────────────────────────────────────────────────────────────


def _faction_of(agent_id: str) -> str:
    """Extract faction prefix from agent ID (e.g. 'GOV-1' -> 'GOV')."""
    return agent_id.split("-")[0] if "-" in agent_id else agent_id


class StimulusEvaluator:
    """Multi-criteria utility evaluator for incoming discourse events.

    Given a NewArgumentEvent, computes how valuable it would be for a
    specific agent to respond to that stimulus. The agent then picks
    the highest-utility stimulus (or a proactive action) and the deficit
    gates whether it actually acts.

    Criteria (weighted):
      1. Faction relevance   (w=0.30) -- does this attack my faction?
      2. Target centrality   (w=0.20) -- is the attacked node structurally central?
      3. Strategic memory    (w=0.15) -- do I have relevant knowledge to counter?
      4. Novelty             (w=0.20) -- has an ally already responded?
      5. Unanswered pressure (w=0.15) -- is this claim uncontested in the graph?
    """

    W_FACTION    = 0.30
    W_CENTRALITY = 0.20
    W_MEMORY     = 0.15
    W_NOVELTY    = 0.20
    W_PRESSURE   = 0.15

    def evaluate(
        self,
        event: "NewArgumentEvent",
        agent_id: str,
        memory: AgentMemory,
        graph: Any,
    ) -> float:
        """Compute expected utility of acting on this stimulus."""
        my_faction = _faction_of(agent_id)

        # 1. Faction relevance
        faction_score = 0.0
        if event.targets_faction and event.targets_faction == my_faction:
            faction_score = 1.0
        elif event.faction != my_faction:
            faction_score = 0.5

        # 2. Target centrality (betweenness of the attacked node)
        centrality_score = 0.0
        if event.target_node_id and hasattr(graph, 'graph'):
            try:
                import networkx as nx
                bc = nx.betweenness_centrality(graph.graph)
                centrality_score = bc.get(event.target_node_id, 0.0)
            except Exception:
                pass

        # 3. Strategic memory -- semantic relevance of stored knowledge
        memory_score = 0.0
        if event.claim:
            relevant = memory.search_relevant(event.claim, "semantic", limit=2)
            memory_score = min(1.0, len(relevant) * 0.5)

        # 4. Novelty -- has an ally already responded to the target?
        novelty_score = 1.0
        if event.node_id and hasattr(graph, 'graph'):
            g = graph.graph
            attackers = list(g.predecessors(event.node_id)) if g.has_node(event.node_id) else []
            ally_responded = any(
                _faction_of(g.nodes[a].get("agent_id", "")) == my_faction
                for a in attackers if a in g.nodes
            )
            if ally_responded:
                novelty_score = 0.2

        # 5. Unanswered pressure -- is this node uncontested?
        pressure_score = 0.0
        if event.node_id and hasattr(graph, 'graph'):
            g = graph.graph
            if g.has_node(event.node_id):
                incoming = g.in_degree(event.node_id)
                pressure_score = 1.0 if incoming == 0 else max(0.0, 1.0 - incoming * 0.3)

        utility = (
            self.W_FACTION * faction_score
            + self.W_CENTRALITY * centrality_score
            + self.W_MEMORY * memory_score
            + self.W_NOVELTY * novelty_score
            + self.W_PRESSURE * pressure_score
        )
        return round(utility, 4)

    def proactive_utility(
        self,
        action: ActionType,
        drive: EpistemicDrive,
        memory: AgentMemory,
        graph_node_count: int,
    ) -> float:
        """Utility of a proactive (non-stimulus) action.

        DEBATE gets high proactive utility when the graph is sparse (bootstrap
        phase) or when the agent has strong debate quality history.
        SEARCH and READ remain auxiliary -- they help prepare for future debates
        but should not dominate during bootstrap.
        """
        p_act = drive.get_activation_probability()
        if action == "DEBATE":
            avg_quality = memory.recent_avg_delta_phi()
            quality_factor = max(0.3, avg_quality / 0.5)
            # Bootstrap bonus: high when graph is empty, drops as graph grows
            bootstrap = max(0.2, 1.0 - graph_node_count / 8.0)
            return round(p_act * (quality_factor + bootstrap) * 0.5, 4)
        elif action == "SEARCH":
            avg_quality = memory.recent_avg_delta_phi()
            quality_deficit = max(0.0, 1.0 - avg_quality / 0.15)
            novelty = 1.0 - memory.semantic_density()
            return round(p_act * quality_deficit * novelty * 0.3, 4)
        elif action == "READ":
            sparsity = max(0.0, 0.2 - memory.semantic_density())
            return round(p_act * (sparsity / 0.2) * 0.2, 4)
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Search provider (Tavily or fallback)
# ──────────────────────────────────────────────────────────────────────────────

_tavily_client: object | None = None
_tavily_checked: bool = False


def _get_tavily_client():
    """Lazy-init the Tavily client.  Returns None if unavailable."""
    global _tavily_client, _tavily_checked
    if _tavily_checked:
        return _tavily_client
    _tavily_checked = True
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.info("TAVILY_API_KEY not set — using fallback knowledge pool")
        return None
    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]
        _tavily_client = TavilyClient(api_key=api_key)
        logger.info("Tavily web search enabled")
    except ImportError:
        logger.warning("tavily-python not installed — using fallback pool")
    return _tavily_client


def _search_tavily(memory: AgentMemory) -> tuple[str, str] | None:
    """Run a real web search via Tavily and return (concept, summary)."""
    client = _get_tavily_client()
    if client is None:
        return None

    query_idx = len(memory.semantic) % len(_SEARCH_QUERIES)
    query = _SEARCH_QUERIES[query_idx]

    try:
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=True,
        )
        answer = response.get("answer", "")
        if not answer:
            results = response.get("results", [])
            if results:
                answer = results[0].get("content", "")[:300]
        if answer:
            concept = f"web_{query_idx}_{len(memory.semantic)}"
            return concept, answer[:400]
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)

    return None


def _search_fallback(memory: AgentMemory) -> tuple[str, str] | None:
    """Draw the next unused fact from the static fallback pool."""
    for concept, fact in _FALLBACK_POOL:
        if concept not in memory.semantic:
            return concept, fact
    return None


def get_search_result(memory: AgentMemory) -> tuple[str, str] | None:
    """Return a (concept, fact) pair for the agent.

    Uses Tavily web search when TAVILY_API_KEY is set; otherwise falls
    back to a static knowledge pool.
    """
    result = _search_tavily(memory)
    if result is not None:
        return result
    return _search_fallback(memory)
