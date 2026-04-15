"""
UtilitySelector: chooses the action that best satisfies the agent's homeostatic drive.

Action space: DEBATE | SEARCH | READ | PASS

Utility functions
─────────────────
  U(DEBATE) — high when δ > θ, graph has attack targets, and recent Δφ* was positive
  U(SEARCH) — high when δ > θ but recent arguments were low-quality (agent needs info)
  U(READ)   — high when δ > θ but semantic memory is sparse (agent lacks background)
  U(PASS)   — 1 − sigmoid(δ): natural resting pressure when the agent is sated

The winning action is argmax over these scores.

SEARCH provider
───────────────
Uses the Tavily Search API for real web search.  When TAVILY_API_KEY is not
set, falls back to a small static knowledge pool so the simulation still
runs offline.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from langclaw.homeostasis import EpistemicDrive
from langclaw.memory import AgentMemory

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
