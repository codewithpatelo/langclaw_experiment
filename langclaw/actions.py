"""Action selection and stimulus evaluation for LangClaw agents.

Two systems coexist:
  UtilitySelector  -- legacy action-type scorer (used by baseline / fallback).
  StimulusEvaluator -- per-event multi-criteria utility (AOP redesign).

SEARCH provider
───────────────
Three-tier search with guaranteed data return:
  1. Tavily  (real web search, requires TAVILY_API_KEY)
  2. DuckDuckGo (free web search, no key needed)
  3. Static knowledge pool (curated facts, recycled when exhausted)

At least one tier always succeeds — agents never get empty research.
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
# Static knowledge pool (tier 3 fallback — always available)
# ──────────────────────────────────────────────────────────────────────────────
_FALLBACK_POOL: list[tuple[str, str]] = [
    # Macroeconomic indicators
    ("PIB_caida", "El PIB cayó un 4.2% en el último trimestre según datos del banco central."),
    ("desempleo", "La tasa de desempleo alcanzó el 18.7%, la más alta en 15 años."),
    ("inflacion", "La inflación acumulada del año supera el 120%, erosionando el salario real."),
    ("deuda_publica", "La deuda pública representa el 89% del PIB, con vencimientos críticos en 6 meses."),
    ("reservas", "Las reservas internacionales cayeron a USD 2.100M, mínimo desde 2005."),
    ("riesgo_pais", "El riesgo país alcanzó 2.340 puntos básicos, cerrando el acceso a mercados internacionales."),
    ("tipo_cambio", "La brecha cambiaria entre el dólar oficial y el paralelo supera el 150%."),
    ("inversion_extranjera", "La inversión extranjera directa cayó un 67% en comparación con el promedio de la década anterior."),
    # Social indicators
    ("pobreza", "El índice de pobreza aumentó del 28% al 41% en los últimos tres años."),
    ("indigencia", "La indigencia pasó del 6% al 12%, duplicándose en apenas dos años según INDEC."),
    ("salud", "El 34% de los hospitales públicos reportan desabastecimiento de medicamentos esenciales."),
    ("mortalidad_infantil", "La mortalidad infantil subió a 11.2 por mil nacidos vivos, revirtiendo una década de mejora."),
    ("desnutricion", "El 16% de niños menores de 5 años presenta desnutrición crónica según UNICEF."),
    ("emigracion", "Se estima que 480.000 ciudadanos emigraron en el último año, la cifra más alta registrada."),
    ("educacion_desercion", "La deserción escolar secundaria alcanzó el 38%, concentrada en quintiles de menor ingreso."),
    ("vivienda", "El déficit habitacional alcanza 3.5 millones de hogares, un 28% de la población."),
    # Fiscal & governance
    ("gasto_social", "El gasto social fue recortado un 23% en términos reales durante la gestión actual."),
    ("presupuesto_educacion", "El presupuesto educativo cayó al 3.1% del PIB, por debajo del mínimo legal del 6%."),
    ("presupuesto_ciencia", "El presupuesto para ciencia y tecnología se redujo al 0.22% del PIB, el nivel más bajo en 20 años."),
    ("subsidios", "Los subsidios a tarifas de servicios públicos representan el 4.1% del PIB, financiados con emisión monetaria."),
    ("recaudacion", "La presión tributaria efectiva es del 25% del PIB, pero la evasión se estima en 35% de lo potencial."),
    # Corruption & institutions
    ("corrupcion", "El índice de percepción de corrupción cayó 12 puntos en el último informe de Transparencia Internacional."),
    ("justicia_lentitud", "El 72% de las causas penales por corrupción llevan más de 5 años sin resolución."),
    ("libertad_prensa", "El país descendió 18 posiciones en el ranking de Reporteros Sin Fronteras en el último bienio."),
    ("independencia_judicial", "El 61% de los ciudadanos percibe al poder judicial como dependiente del ejecutivo según Latinobarómetro."),
    ("contrataciones", "El 43% de las contrataciones públicas se realizaron por adjudicación directa sin licitación competitiva."),
    # Infrastructure & productivity
    ("infraestructura_vial", "Solo el 12% de las rutas nacionales están en buen estado según la Dirección Nacional de Vialidad."),
    ("energia", "Los cortes de energía aumentaron un 340% respecto al año anterior, afectando la producción industrial."),
    ("conectividad", "El 28% de la población rural no tiene acceso a internet de banda ancha."),
    ("productividad", "La productividad laboral cayó un 15% en los últimos 5 años según la OIT."),
    # International comparisons
    ("comparacion_chile", "Chile mantiene una tasa de pobreza del 10.8% con un gasto social del 14% del PIB, versus 41% de pobreza con 8% de gasto social en nuestro caso."),
    ("comparacion_uruguay", "Uruguay destina el 5.1% del PIB a educación y tiene una tasa de finalización secundaria del 72%, versus 3.1% y 62% respectivamente."),
    ("comparacion_ocde", "El promedio OCDE de inversión en I+D es 2.7% del PIB; el país invierte 0.22%."),
    ("IDH", "El Índice de Desarrollo Humano del país cayó de 0.845 a 0.791 en la última década, saliendo del grupo de desarrollo humano muy alto."),
    # Government defense arguments
    ("programa_social_1", "El programa Alimentar Futuro alcanzó 2.3 millones de beneficiarios, reduciendo la inseguridad alimentaria aguda en 8 puntos."),
    ("obra_publica", "Se inauguraron 1.200 km de rutas pavimentadas y 45 hospitales modulares en zonas rurales durante la gestión."),
    ("empleo_registrado", "El empleo registrado en PyMEs creció un 4.2% interanual gracias al programa REPRO III."),
    ("exportaciones", "Las exportaciones agroindustriales alcanzaron un récord de USD 42.000M, un 18% más que el año anterior."),
    ("acuerdo_deuda", "Se reestructuraron USD 65.000M de deuda con acreedores privados, posponiendo vencimientos hasta 2030."),
    ("plan_conectar", "El plan Conectar Igualdad distribuyó 800.000 notebooks a estudiantes de escuelas públicas."),
    ("vacunacion", "La campaña de vacunación alcanzó al 87% de la población objetivo con esquema completo."),
    ("seguridad_social", "La cobertura previsional se amplió al 96% de los mayores de 65 años mediante moratorias."),
    # Historical & structural
    ("ciclos_economicos", "El país ha experimentado 7 recesiones en los últimos 20 años, con un patrón de stop-and-go crónico."),
    ("estructura_productiva", "La participación industrial en el PIB cayó del 22% al 14% en dos décadas, profundizando la reprimarización."),
    ("informalidad", "La economía informal representa el 45% del empleo total, limitando la base imponible y la protección social."),
    ("concentracion_riqueza", "El decil más rico concentra el 32% del ingreso total, mientras el decil más pobre recibe el 1.8%."),
    ("fuga_capitales", "La fuga de capitales acumulada en la última década supera los USD 90.000M según estimaciones del BCRA."),
    ("pbi_per_capita", "El PIB per cápita en dólares constantes es hoy un 18% inferior al de 2011."),
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
    """Extract faction prefix from agent ID (e.g. 'GOV-S1' -> 'GOV')."""
    return agent_id.split("-")[0] if "-" in agent_id else agent_id


class StimulusEvaluator:
    """Multi-criteria utility evaluator for incoming discourse events.

    Given a NewArgumentEvent, computes how valuable it would be for a
    specific agent to respond to that stimulus. The agent then picks
    the highest-utility stimulus (or a proactive action) and the deficit
    gates whether it actually acts.

    Criteria:
      1. Faction relevance   -- does this attack my faction?
      2. Target centrality   -- is the attacked node structurally central?
      3. Strategic memory    -- do I have relevant knowledge to counter?
      4. Novelty             -- has an ally already responded?
      5. Unanswered pressure -- is this claim uncontested in the graph?

    Default weights are calibrated via ablation study on micro-simulation
    pilot data (see calibrate_hyperparams.py).  Weights are passed at
    construction time to support reproducible calibration.
    """

    def __init__(
        self,
        w_faction: float = 0.20,
        w_centrality: float = 0.20,
        w_memory: float = 0.20,
        w_novelty: float = 0.20,
        w_pressure: float = 0.20,
    ) -> None:
        total = w_faction + w_centrality + w_memory + w_novelty + w_pressure
        self.W_FACTION = w_faction / total
        self.W_CENTRALITY = w_centrality / total
        self.W_MEMORY = w_memory / total
        self.W_NOVELTY = w_novelty / total
        self.W_PRESSURE = w_pressure / total

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
_tavily_exhausted: bool = False


def _get_tavily_client():
    """Lazy-init the Tavily client.  Returns None if unavailable."""
    global _tavily_client, _tavily_checked
    if _tavily_checked:
        return _tavily_client
    _tavily_checked = True
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.info("TAVILY_API_KEY not set — will try DuckDuckGo")
        return None
    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]
        _tavily_client = TavilyClient(api_key=api_key)
        logger.info("Tavily web search enabled")
    except ImportError:
        logger.warning("tavily-python not installed — will try DuckDuckGo")
    return _tavily_client


def _search_tavily(memory: AgentMemory) -> tuple[str, str] | None:
    """Tier 1: Tavily real web search."""
    global _tavily_exhausted
    if _tavily_exhausted:
        return None

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
        exc_str = str(exc)
        if "usage limit" in exc_str.lower() or "rate" in exc_str.lower():
            logger.warning("Tavily quota exhausted — switching to DuckDuckGo: %s", exc)
            _tavily_exhausted = True
        else:
            logger.warning("Tavily search failed: %s", exc)

    return None


def _search_duckduckgo(memory: AgentMemory) -> tuple[str, str] | None:
    """Tier 2: DuckDuckGo free web search (no API key needed)."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-untyped]
        except ImportError:
            return None

    query_idx = len(memory.semantic) % len(_SEARCH_QUERIES)
    query = _SEARCH_QUERIES[query_idx]

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if results:
            best = results[0]
            body = best.get("body", "") or best.get("title", "")
            if len(results) > 1:
                body += " " + (results[1].get("body", "") or "")
            body = body.strip()[:400]
            if body:
                concept = f"ddg_{query_idx}_{len(memory.semantic)}"
                return concept, body
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)

    return None


_fallback_cycle_counter: int = 0


def _search_fallback(memory: AgentMemory) -> tuple[str, str]:
    """Tier 3: static knowledge pool with recycling.

    Always returns data.  When all facts have been seen, cycles through
    them again with unique concept keys so memory accepts them.
    """
    global _fallback_cycle_counter

    for concept, fact in _FALLBACK_POOL:
        if concept not in memory.semantic:
            return concept, fact

    idx = _fallback_cycle_counter % len(_FALLBACK_POOL)
    _fallback_cycle_counter += 1
    _, fact = _FALLBACK_POOL[idx]
    concept = f"pool_r{_fallback_cycle_counter}_{idx}"
    return concept, fact


def get_search_result(memory: AgentMemory) -> tuple[str, str]:
    """Return a (concept, fact) pair — guaranteed non-None.

    Three-tier fallback:
      1. Tavily (real web search, if key available and quota not exhausted)
      2. DuckDuckGo (free web search, no key needed)
      3. Static knowledge pool (curated, recycled when exhausted)
    """
    result = _search_tavily(memory)
    if result is not None:
        return result

    result = _search_duckduckgo(memory)
    if result is not None:
        return result

    return _search_fallback(memory)
