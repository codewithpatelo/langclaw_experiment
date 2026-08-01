"""Symbolic argumentation layer — quality signal g and AAF metrics.

Implements a directed argument graph where nodes are claims and edges
represent logical attacks (undercut / rebuttal).

Quality signal g (calculate_quality_signal)
--------------------------------------------
Operationalises A3 (Quality gate) from the Pro-Action operator Γ.
The signal g ∈ [0, 1] measures whether a new argument engages with
the content it attacks and contributes novel content, rather than
repeating or producing disconnected monologue:

    g = w_e · engagement(claim, target_claim)
      + w_n · novelty(claim, own_history)
      + w_d · diversity(target_node, graph)

  1. Engagement: token overlap between the new claim and the target claim.
     Detects whether the agent addresses the specific content of the
     argument it attacks.  Low engagement = disconnected monologue.
  2. Novelty: 1 − max overlap with the agent's own previous claims.
     Detects repetition.  Low novelty = the agent is repeating itself.
  3. Diversity (retained from original design): ratio of distinct agents
     among the target's neighbours.  Low diversity = monopolisation of
     the discourse by few agents.

This replaces the original structural proxy (betweenness + cycle + diversity)
which was circular: it measured graph position, not argumentative quality,
yet fed back into the homeostatic satiation loop.  The new signal closes
the loop on content-level anti-trivialisation, directly addressing the
reviewer concern about metric circularity.

AAF metrics (enriched Dung 1995)
-------------------------------
Grounded-extension-based evaluation with attack-type distinction:
  - Rebuttal: standard AAF defeat (target falls, its attacks stay active)
  - Undercut: defeats target AND disables its outgoing attacks (broken inference)
  - defeat_cycle_count(): |SCC_{>1}| via Tarjan's SCC (genuine dialectical tension)
  - acceptance_ratio(): α = |grounded extension| / |total nodes|
  - dialectical_completeness(): δ = |nodes addressed by GE| / |total nodes|
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Token overlap utility for engagement / novelty computation
# ──────────────────────────────────────────────────────────────────────────────

_STOPWORDS_ES = frozenset({
    "que", "de", "la", "el", "en", "y", "a", "los", "del", "las", "por",
    "con", "para", "se", "su", "es", "una", "un", "no", "como", "pero",
    "mas", "fue", "han", "hay", "esta", "este", "estos", "estas", "eso",
    "esa", "ese", "o", "si", "ya", "al", "lo", "le", "les", "te", "me",
    "mi", "tu", "sus", "nuestra", "nuestro", "ha", "han", "ser", "son",
    "tiene", "tienen", "puede", "pueden", "esto", "tambien", "ademas",
})

_STOPWORDS_EN = frozenset({
    "the", "that", "this", "these", "those", "with", "from", "have",
    "has", "are", "was", "were", "been", "being", "will", "would",
    "could", "should", "shall", "may", "might", "must", "can", "cannot",
    "not", "but", "and", "for", "you", "your", "they", "their", "them",
    "his", "her", "she", "him", "its", "our", "ours", "also", "than",
})

_MIN_TOKEN_LEN = 4


def _significant_tokens(text: str) -> set[str]:
    """Extract significant tokens: lowercase, >4 chars, no stopwords."""
    tokens = set()
    for word in text.lower().split():
        cleaned = word.strip(".,;:!?\"'()[]{}¿¡—–-")
        if len(cleaned) >= _MIN_TOKEN_LEN and cleaned not in _STOPWORDS_ES and cleaned not in _STOPWORDS_EN:
            tokens.add(cleaned)
    return tokens


def _token_overlap(text_a: str, text_b: str) -> float:
    """Jaccard-style overlap ratio: |A ∩ B| / |B|.

    Returns 0.0 if text_b has no significant tokens.
    Measures what fraction of the target's significant tokens
    appear in the new claim.
    """
    tokens_a = _significant_tokens(text_a)
    tokens_b = _significant_tokens(text_b)
    if not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_b)


class ArgumentGraph:
    """Directed graph of argumentative claims and attacks.

    Each node stores:
        - agent_id : who produced the claim
        - claim    : the textual content
        - tick     : simulation tick when it was added

    Each edge stores:
        - attack_type : "undercut" | "rebuttal"
    """

    def __init__(self, connectivity_fix: bool = True) -> None:
        self._graph = nx.DiGraph()
        self._node_order: list[str] = []
        self._lock = asyncio.Lock()  # guards concurrent writes from async agents
        self._connectivity_fix = connectivity_fix

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    async def add_argument_async(
        self,
        agent_id: str,
        claim: str,
        target_node_id: str | None = None,
        attack_type: str | None = None,
        tick: int = 0,
        node_id: str | None = None,
    ) -> str:
        """Thread-safe async version of add_argument (used by async agents)."""
        async with self._lock:
            return self._add_argument_inner(
                agent_id, claim, target_node_id, attack_type, tick, node_id=node_id
            )

    def add_argument(
        self,
        agent_id: str,
        claim: str,
        target_node_id: str | None = None,
        attack_type: str | None = None,
        tick: int = 0,
        node_id: str | None = None,
    ) -> str:
        """Add a claim node and optionally an attack edge.

        Returns the generated ``node_id``.
        """
        return self._add_argument_inner(
            agent_id, claim, target_node_id, attack_type, tick, node_id=node_id
        )

    def _resolve_target(
        self, target_node_id: str | None, current_node_id: str | None = None
    ) -> str | None:
        """Resolve a target_node_id to an existing graph node.

        If the exact ID exists, return it.  Otherwise attempt a fuzzy
        match (prefix or suffix) to handle LLM truncation/mangling.
        If no match is found and other nodes exist, pick the most recent
        one to prevent isolated nodes.  Returns None only when no other
        node exists (no possible target).

        When ``connectivity_fix`` is False (no-fix ablation), fuzzy
        matching and fallback are skipped — hallucinated targets create
        isolated nodes with no edge, preserving the pre-fix behaviour.
        """
        if not target_node_id:
            return None

        if self._graph.has_node(target_node_id) and target_node_id != current_node_id:
            return target_node_id

        if not self._connectivity_fix:
            return None

        existing = [n for n in self._graph.nodes if n != current_node_id]
        if not existing:
            return None

        # Fuzzy match: prefix (LLM truncated the UUID suffix)
        prefix_matches = [n for n in existing if n.startswith(target_node_id[:10])]
        if len(prefix_matches) == 1:
            logger.warning(
                "Fuzzy prefix match: '%s' -> '%s'", target_node_id, prefix_matches[0]
            )
            return prefix_matches[0]

        # Fuzzy match: suffix (LLM mangled the agent prefix)
        suffix_matches = [n for n in existing if n.endswith(target_node_id[-8:])]
        if len(suffix_matches) == 1:
            logger.warning(
                "Fuzzy suffix match: '%s' -> '%s'", target_node_id, suffix_matches[0]
            )
            return suffix_matches[0]

        # Fallback: pick the most recent existing node to maintain connectivity
        fallback = existing[-1]
        logger.warning(
            "Target node '%s' not found, falling back to '%s'", target_node_id, fallback
        )
        return fallback

    def _add_argument_inner(
        self,
        agent_id: str,
        claim: str,
        target_node_id: str | None = None,
        attack_type: str | None = None,
        tick: int = 0,
        node_id: str | None = None,
    ) -> str:
        node_id = node_id or f"{agent_id}_{uuid.uuid4().hex[:8]}"
        self._graph.add_node(node_id, agent_id=agent_id, claim=claim, tick=tick)
        self._node_order.append(node_id)

        resolved_target = self._resolve_target(target_node_id, current_node_id=node_id)
        if resolved_target:
            self._graph.add_edge(
                node_id,
                resolved_target,
                attack_type=attack_type or "rebuttal",
            )

        return node_id

    def to_checkpoint(self) -> dict[str, Any]:
        """Serialize the argument graph for benchmark resume."""
        return {
            "nodes": [
                {"node_id": node_id, **dict(self._graph.nodes[node_id])}
                for node_id in self._node_order
                if self._graph.has_node(node_id)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    **dict(data),
                }
                for source, target, data in self._graph.edges(data=True)
            ],
            "node_order": list(self._node_order),
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> "ArgumentGraph":
        """Restore a serialized argument graph."""
        graph = cls()
        graph._graph.clear()
        graph._node_order = []

        for node in payload.get("nodes", []):
            node_id = node["node_id"]
            graph._graph.add_node(
                node_id,
                agent_id=node.get("agent_id", ""),
                claim=node.get("claim", ""),
                tick=node.get("tick", 0),
            )

        for edge in payload.get("edges", []):
            source = edge["source"]
            target = edge["target"]
            if graph._graph.has_node(source) and graph._graph.has_node(target):
                graph._graph.add_edge(
                    source,
                    target,
                    attack_type=edge.get("attack_type", "rebuttal"),
                )

        node_order = payload.get("node_order")
        if node_order:
            graph._node_order = [n for n in node_order if graph._graph.has_node(n)]
        else:
            graph._node_order = list(graph._graph.nodes())

        return graph

    def calculate_quality_signal(
        self,
        node_id: str,
        agent_claim_history: list[str] | None = None,
    ) -> float:
        """Compute the quality signal g for a newly added node (A3 gate).

        This is the feedback signal that closes the homeostatic loop in
        Γ(n=1): it measures whether the agent's contribution engages with
        the content it attacks and adds novelty, rather than repeating or
        producing disconnected monologue.

        Components (each in [0, 1]):
            1. Engagement — token overlap between the new claim and the
               target claim.  Detects content-level addressal.
            2. Novelty — 1 − max overlap with the agent's own previous
               claims.  Detects repetition / trivialisation.
            3. Diversity — ratio of distinct agents among the target's
               neighbours.  Detects monopolisation of discourse.

        Isolated nodes (no outgoing attack) return 0.0: by A3, an action
        that does not engage with the discourse provides no satiation.

        Parameters
        ----------
        node_id:
            The newly added node to evaluate.
        agent_claim_history:
            List of this agent's previous claim texts (for novelty
            computation).  If empty or None, novelty defaults to 1.0
            (first claim is always novel).
        """
        if node_id not in self._graph:
            return 0.0

        successors = list(self._graph.successors(node_id))
        if not successors:
            return 0.0

        target = successors[0]
        new_claim = self._graph.nodes[node_id].get("claim", "")
        target_claim = self._graph.nodes[target].get("claim", "")

        # --- 1. Engagement: token overlap with target claim ---
        engagement = _token_overlap(new_claim, target_claim)

        # --- 2. Novelty: 1 - max overlap with own history ---
        if agent_claim_history:
            max_self_overlap = max(
                _token_overlap(new_claim, prev) for prev in agent_claim_history
            )
            novelty = 1.0 - max_self_overlap
        else:
            novelty = 1.0

        # --- 3. Diversity: distinct agents among target's neighbours ---
        neighbours = set(self._graph.predecessors(target)) | set(
            self._graph.successors(target)
        )
        neighbours.add(node_id)
        unique_agents = {
            self._graph.nodes[n].get("agent_id") for n in neighbours if n in self._graph
        }
        total_agents_in_graph = len(
            {d.get("agent_id") for _, d in self._graph.nodes(data=True)}
        )
        diversity_score = (
            len(unique_agents) / total_agents_in_graph if total_agents_in_graph else 0.0
        )

        w_e, w_n, w_d = 1/3, 1/3, 1/3
        g = w_e * engagement + w_n * novelty + w_d * diversity_score
        return min(1.0, max(0.0, g))

    # Backward-compatible alias
    def calculate_phi_star_proxy(
        self,
        node_id: str,
        agent_claim_history: list[str] | None = None,
    ) -> float:
        """Deprecated alias for calculate_quality_signal."""
        return self.calculate_quality_signal(node_id, agent_claim_history)

    # ──────────────────────────────────────────────────────────────────────────
    # AAF metrics (Dung 1995)
    # ──────────────────────────────────────────────────────────────────────────

    def _grounded_extension(self) -> set[str]:
        """Compute the grounded extension with undercut/rebuttal semantics.

        Enriches Dung's AAF with attack-type distinction:
        - Rebuttal: standard AAF defeat. The target is defeated but its
          outgoing attacks remain active (the conclusion is wrong, but
          the argument can still attack others).
        - Undercut: defeats the target AND disables its outgoing attacks
          (the inference is broken, so the argument cannot defeat others).

        Algorithm: iteratively add to the extension all arguments whose
        active attackers are themselves attacked by the current extension.
        An attack is "active" if the attacker is not undercut-defeated
        (i.e., its inference is not broken by a GE member).

        Runs in polynomial time (O(|A|·|→|) per iteration, at most |A| iterations).
        """
        g = self._graph
        if g.number_of_nodes() == 0:
            return set()

        # Start: all arguments with no attackers are unconditionally in the GE
        ge: set[str] = {n for n in g.nodes() if g.in_degree(n) == 0}

        while True:
            # Arguments defeated by current GE
            defeated: set[str] = set()
            # Nodes whose inference is broken by an undercut from GE
            inference_broken: set[str] = set()

            for s in ge:
                for target in g.successors(s):
                    edge_data = g.edges[s, target]
                    attack_type = edge_data.get("attack_type", "rebuttal")
                    defeated.add(target)
                    if attack_type == "undercut":
                        inference_broken.add(target)

            # Add arguments all of whose *active* attackers are defeated by GE
            # An attacker is "active" if its inference is not broken
            new_members: set[str] = set()
            for n in g.nodes():
                if n in ge or n in defeated:
                    continue
                attackers = set(g.predecessors(n))
                if not attackers:
                    continue
                # Only count attackers whose inference is not broken
                active_attackers = attackers - inference_broken
                if active_attackers and all(a in defeated for a in active_attackers):
                    new_members.add(n)

            if not new_members:
                break
            ge = ge | new_members

        return ge

    def defeat_cycle_count(self) -> int:
        """Count strongly-connected components with more than one node (defeat cycles).

        A non-trivial SCC (|SCC| > 1) indicates a genuine dialectical cycle in the
        attack graph — mutual refutation between arguments. Higher counts suggest
        richer dialectical engagement.

        Uses networkx.strongly_connected_components (Tarjan's algorithm, O(|A|+|→|)).
        """
        sccs = list(nx.strongly_connected_components(self._graph))
        return sum(1 for scc in sccs if len(scc) > 1)

    def acceptance_ratio(self) -> float:
        """Fraction of arguments in the grounded extension.

        α = |GE| / |A|  (Dung 1995)

        α = 1.0: all arguments are epistemically undefeated.
        α = 0.0: all arguments are contested (no stable grounded truth).
        Returns 0.0 for an empty graph.
        """
        n = self._graph.number_of_nodes()
        if n == 0:
            return 0.0
        ge = self._grounded_extension()
        return len(ge) / n

    def dialectical_completeness(self) -> float:
        """Fraction of arguments addressed by the grounded extension.

        δ = |{x : x ∈ GE or x is attacked by GE}| / |A|

        An argument is "addressed" when the grounded extension either accepts it
        or defeats it.  δ = 0 means the discourse is fully indeterminate; δ = 1
        means the GE has a position on every claim.
        Returns 0.0 for an empty graph.
        """
        g = self._graph
        n = g.number_of_nodes()
        if n == 0:
            return 0.0

        ge = self._grounded_extension()
        # Arguments defeated by GE
        defeated_by_ge: set[str] = set()
        for s in ge:
            defeated_by_ge.update(g.successors(s))

        addressed = ge | defeated_by_ge
        return len(addressed) / n

    def get_recent_context(self, last_n: int = 5) -> str:
        """Return a textual summary of the last *n* arguments for prompt injection.

        Shows both outgoing attacks (A attacks B) and incoming attacks
        (who is attacking A), enabling agents to identify undefended claims.
        """
        recent = self._node_order[-last_n:]
        if not recent:
            return "No arguments have been made yet."

        lines: list[str] = []
        for nid in recent:
            data = self._graph.nodes[nid]
            targets = list(self._graph.successors(nid))
            target_info = ""
            if targets:
                edge_data = self._graph.edges[nid, targets[0]]
                target_info = f" --[{edge_data.get('attack_type', '?')}]--> {targets[0]}"
            attackers = list(self._graph.predecessors(nid))
            attacked_by = ""
            if attackers:
                attacked_by = f" (ATTACKED BY: {', '.join(attackers)})"
            lines.append(
                f"[{nid}] ({data['agent_id']}): \"{data['claim']}\"{target_info}{attacked_by}"
            )
        return "\n".join(lines)

    def get_state_summary(self) -> dict[str, Any]:
        """Graph statistics for the dashboard."""
        g = self._graph
        return {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "density": nx.density(g) if g.number_of_nodes() > 1 else 0.0,
            "components": nx.number_weakly_connected_components(g) if g.number_of_nodes() else 0,
        }

    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Return all nodes with attributes (for visualisation)."""
        return [
            {"id": nid, **data} for nid, data in self._graph.nodes(data=True)
        ]

    def get_all_edges(self) -> list[dict[str, Any]]:
        """Return all edges with attributes (for visualisation)."""
        return [
            {"source": u, "target": v, **data}
            for u, v, data in self._graph.edges(data=True)
        ]

    def valid_target_ids(self) -> list[str]:
        """Return node IDs available as attack targets."""
        return list(self._graph.nodes)

    def get_undefended_attacks(self, faction_prefix: str) -> list[dict[str, str]]:
        """Find opponent attacks on this faction's claims that have no counter-attack.

        Returns a list of dicts with 'attacker_node', 'attacked_node', and
        'attacker_claim' — these are high-priority targets for counter-attack,
        which can produce defeat cycles (mutual refutation).
        """
        results = []
        for nid in self._graph.nodes:
            if not nid.startswith(faction_prefix):
                attackers = list(self._graph.predecessors(nid))
                continue
            attackers = list(self._graph.predecessors(nid))
            for attacker_nid in attackers:
                if attacker_nid.startswith(faction_prefix):
                    continue
                counter_attacks = [
                    pred for pred in self._graph.predecessors(attacker_nid)
                    if pred.startswith(faction_prefix)
                ]
                if not counter_attacks:
                    attacker_data = self._graph.nodes[attacker_nid]
                    results.append({
                        "attacker_node": attacker_nid,
                        "attacked_node": nid,
                        "attacker_claim": attacker_data.get("claim", "")[:120],
                    })
        return results
