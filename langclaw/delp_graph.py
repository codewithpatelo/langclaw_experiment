"""Symbolic argumentation layer and IIT (Φ*) proxy.

Implements a directed argument graph where nodes are claims and edges
represent logical attacks (undercut / rebuttal).  The ``calculate_phi_star_proxy``
method approximates Integrated Information Theory's Φ* using tractable
network-theoretic features:

    Φ* ≈ w_c · C_betweenness(target) + w_cycle · has_cycle + w_div · agent_diversity

This is explicitly a *proxy* — full IIT is computationally intractable for
graphs of this size in real-time.  The proxy captures the intuition that
an argument is informationally valuable when it engages central claims,
creates dialectical cycles, and bridges perspectives from different agents.
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
    ) -> str:
        """Thread-safe async version of add_argument (used by async agents)."""
        async with self._lock:
            return self._add_argument_inner(
                agent_id, claim, target_node_id, attack_type, tick
            )

    def add_argument(
        self,
        agent_id: str,
        claim: str,
        target_node_id: str | None = None,
        attack_type: str | None = None,
        tick: int = 0,
    ) -> str:
        """Add a claim node and optionally an attack edge.

        Returns the generated ``node_id``.
        """
        return self._add_argument_inner(agent_id, claim, target_node_id, attack_type, tick)

    def _add_argument_inner(
        self,
        agent_id: str,
        claim: str,
        target_node_id: str | None = None,
        attack_type: str | None = None,
        tick: int = 0,
    ) -> str:
        node_id = f"{agent_id}_{uuid.uuid4().hex[:8]}"
        self._graph.add_node(node_id, agent_id=agent_id, claim=claim, tick=tick)
        self._node_order.append(node_id)

        if target_node_id and self._graph.has_node(target_node_id):
            self._graph.add_edge(
                node_id,
                target_node_id,
                attack_type=attack_type or "rebuttal",
            )

        return node_id

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

        w_c, w_cycle, w_div = 0.35, 0.35, 0.30
        phi = w_c * centrality_score + w_cycle * cycle_bonus + w_div * diversity_score
        return min(1.0, max(0.0, phi))

    def get_recent_context(self, last_n: int = 5) -> str:
        """Return a textual summary of the last *n* arguments for prompt injection."""
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
            lines.append(
                f"[{nid}] ({data['agent_id']}): \"{data['claim']}\"{target_info}"
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
