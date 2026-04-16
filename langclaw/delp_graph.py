"""Symbolic argumentation layer — network-theoretic quality proxy (Δφ*) and AAF metrics.

Implements a directed argument graph where nodes are claims and edges
represent logical attacks (undercut / rebuttal).

Δφ* quality proxy (calculate_phi_star_proxy)
--------------------------------------------
A network-theoretic proxy that operationalises three structural features
of argumentative integration:

    Δφ* ≈ w_c · C_betweenness(target) + w_cycle · 1_cycle + w_div · D_diversity

  1. Betweenness centrality C_betweenness(target): whether the attacked node is
     structurally central to the discourse.
  2. Cycle bonus 1_cycle: whether the new attack creates a dialectical cycle,
     rewarding genuine back-and-forth over parallel monologue.
  3. Cross-faction diversity D: whether the target has been engaged by agents
     from multiple factions.

These features are conceptually related to — though formally distinct from —
informational integration (Tononi et al.); no formal equivalence is claimed.
Weights (1/3, 1/3, 1/3) use a maximum-entropy equal-importance prior (Jaynes
1957); no domain evidence favours one component over another a priori.

AAF metrics (Dung 1995)
-----------------------
Grounded-extension-based evaluation of the argument graph:
  - defeat_cycle_count(): |SCC_{>1}| via Tarjan's SCC (genuine dialectical tension)
  - acceptance_ratio(): α = |grounded extension| / |total nodes|
  - dialectical_completeness(): δ = |nodes addressed by GE| / |total nodes|
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import networkx as nx


class ArgumentGraph:
    """Directed graph of argumentative claims and attacks.

    Each node stores:
        - agent_id : who produced the claim
        - claim    : the textual content
        - tick     : simulation tick when it was added

    Each edge stores:
        - attack_type : "undercut" | "rebuttal"
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._node_order: list[str] = []
        self._lock = asyncio.Lock()  # guards concurrent writes from async agents

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

        if target_node_id and self._graph.has_node(target_node_id):
            self._graph.add_edge(
                node_id,
                target_node_id,
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

    def calculate_phi_star_proxy(self, node_id: str) -> float:
        """Compute an IIT Φ* proxy for the newly added node.

        Components (each in [0, 1]):
            1. Betweenness centrality of the attack target.
            2. Cycle bonus — does the new edge create a refutation cycle?
            3. Agent diversity — ratio of distinct agents connected.

        Isolated nodes (no outgoing attack) return 0.0.
        """
        if node_id not in self._graph:
            return 0.0

        successors = list(self._graph.successors(node_id))
        if not successors:
            return 0.0

        target = successors[0]

        # --- 1. Betweenness centrality of the target ---
        bc = nx.betweenness_centrality(self._graph)
        centrality_score = bc.get(target, 0.0)

        # --- 2. Cycle detection bonus ---
        cycle_bonus = 0.0
        try:
            path = nx.shortest_path(self._graph, target, node_id)
            if len(path) >= 2:
                cycle_bonus = 1.0
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # --- 3. Agent diversity of neighbours ---
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

        w_c, w_cycle, w_div = 1/3, 1/3, 1/3
        phi = w_c * centrality_score + w_cycle * cycle_bonus + w_div * diversity_score
        return min(1.0, max(0.0, phi))

    # ──────────────────────────────────────────────────────────────────────────
    # AAF metrics (Dung 1995)
    # ──────────────────────────────────────────────────────────────────────────

    def _grounded_extension(self) -> set[str]:
        """Compute the grounded extension of the AAF via fixed-point iteration.

        The grounded extension is the unique minimal complete extension (Dung 1995).
        Algorithm: iteratively add to the extension all arguments whose attackers
        are themselves attacked by the current extension, until no change.

        Runs in polynomial time (O(|A|·|→|) per iteration, at most |A| iterations).
        """
        g = self._graph
        if g.number_of_nodes() == 0:
            return set()

        # Start: all arguments with no attackers are unconditionally in the GE
        ge: set[str] = {n for n in g.nodes() if g.in_degree(n) == 0}

        while True:
            # Arguments defeated by current GE (attacked by at least one member)
            defeated: set[str] = set()
            for s in ge:
                defeated.update(g.successors(s))  # s attacks these nodes

            # Add arguments all of whose attackers are defeated by GE
            new_members: set[str] = set()
            for n in g.nodes():
                if n in ge or n in defeated:
                    continue
                attackers = set(g.predecessors(n))
                if attackers and all(a in defeated for a in attackers):
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
