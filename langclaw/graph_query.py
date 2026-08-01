"""Deterministic graph query engine for SEARCH actions.

Replaces web search with reproducible structural analysis over the
NetworkX argument graph. The LLM selects a query by name; the engine
executes it deterministically and returns text results.

All queries are pure functions of the graph state — no API calls,
no randomness, fully reproducible.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def _faction_of(agent_id: str) -> str:
    return agent_id.split("-")[0] if "-" in agent_id else agent_id


def query_undefended_attacks(
    graph: nx.DiGraph, faction: str | None = None
) -> str:
    """Find attacks on the specified faction that have no counter-attack.

    If faction is None, infers the opposing faction from the graph.
    """
    if graph.number_of_nodes() == 0:
        return "No arguments in the graph yet."

    target_faction = faction or ""
    if not target_faction:
        factions = {_faction_of(d.get("agent_id", "")) for _, d in graph.nodes(data=True)}
        if len(factions) == 2:
            target_faction = sorted(factions)[0]
        else:
            return "Cannot determine faction from graph."

    results: list[str] = []
    for nid, data in graph.nodes(data=True):
        if not nid.startswith(target_faction):
            continue
        attackers = list(graph.predecessors(nid))
        for att in attackers:
            if att.startswith(target_faction):
                continue
            counter = [p for p in graph.predecessors(att) if p.startswith(target_faction)]
            if not counter:
                att_data = graph.nodes[att]
                results.append(
                    f"- [{att}] ({att_data.get('agent_id', '?')}) attacks "
                    f"[{nid}]: \"{att_data.get('claim', '')[:120]}\""
                )

    if not results:
        return f"No undefended attacks on {target_faction} faction."
    header = f"Undefended attacks on {target_faction} ({len(results)}):\n"
    return header + "\n".join(results)


def query_weak_opponent_nodes(
    graph: nx.DiGraph, faction: str | None = None, min_centrality: float = 0.0
) -> str:
    """Find opponent nodes with low defense (few or no counter-attacks)."""
    if graph.number_of_nodes() == 0:
        return "No arguments in the graph yet."

    target_faction = faction or ""
    if not target_faction:
        factions = {_faction_of(d.get("agent_id", "")) for _, d in graph.nodes(data=True)}
        if len(factions) == 2:
            target_faction = sorted(factions)[0]
        else:
            return "Cannot determine faction from graph."

    try:
        centrality = nx.betweenness_centrality(graph)
    except Exception:
        centrality = {n: 0.0 for n in graph.nodes()}

    results: list[tuple[float, str, str, str]] = []
    for nid, data in graph.nodes(data=True):
        if not nid.startswith(target_faction):
            continue
        in_deg = graph.in_degree(nid)
        c = centrality.get(nid, 0.0)
        if c >= min_centrality:
            results.append((c, nid, data.get("agent_id", "?"), data.get("claim", "")[:100]))

    results.sort(key=lambda x: x[0])
    if not results:
        return f"No opponent ({target_faction}) nodes found."

    lines = [f"Opponent ({target_faction}) nodes by centrality (ascending):"]
    for c, nid, aid, claim in results[:10]:
        defenders = [p for p in graph.predecessors(nid) if p.startswith(target_faction)]
        defense_status = f"defended by {len(defenders)}" if defenders else "UNDEFENDED"
        lines.append(f"- [{nid}] ({aid}) centrality={c:.3f} {defense_status}: \"{claim}\"")
    return "\n".join(lines)


def query_unattacked_nodes(
    graph: nx.DiGraph, faction: str | None = None, sort_by: str = "centrality"
) -> str:
    """Find nodes of the specified faction that nobody has attacked."""
    if graph.number_of_nodes() == 0:
        return "No arguments in the graph yet."

    target_faction = faction or ""
    if not target_faction:
        factions = {_faction_of(d.get("agent_id", "")) for _, d in graph.nodes(data=True)}
        if len(factions) == 2:
            target_faction = sorted(factions)[0]
        else:
            return "Cannot determine faction from graph."

    unattacked: list[tuple[float, str, str, str]] = []
    try:
        centrality = nx.betweenness_centrality(graph)
    except Exception:
        centrality = {n: 0.0 for n in graph.nodes()}

    for nid, data in graph.nodes(data=True):
        if not nid.startswith(target_faction):
            continue
        if graph.in_degree(nid) == 0:
            c = centrality.get(nid, 0.0)
            unattacked.append((c, nid, data.get("agent_id", "?"), data.get("claim", "")[:100]))

    if not unattacked:
        return f"All {target_faction} nodes have been attacked."

    if sort_by == "centrality":
        unattacked.sort(key=lambda x: x[0], reverse=True)

    lines = [f"Unattacked {target_faction} nodes ({len(unattacked)}):"]
    for c, nid, aid, claim in unattacked[:10]:
        lines.append(f"- [{nid}] ({aid}) centrality={c:.3f}: \"{claim}\"")
    return "\n".join(lines)


def query_attack_chains(
    graph: nx.DiGraph, min_depth: int = 2, faction: str | None = None
) -> str:
    """Find the longest attack chains in the graph."""
    if graph.number_of_nodes() == 0:
        return "No arguments in the graph yet."

    def find_chains(node: str, visited: set[str]) -> list[list[str]]:
        successors = [s for s in graph.successors(node) if s not in visited]
        if not successors:
            return [[node]]
        chains: list[list[str]] = []
        for s in successors:
            for chain in find_chains(s, visited | {node}):
                chains.append([node] + chain)
        return chains

    all_chains: list[list[str]] = []
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            all_chains.extend(find_chains(node, set()))

    all_chains = [c for c in all_chains if len(c) >= min_depth]
    all_chains.sort(key=len, reverse=True)

    if not all_chains:
        return f"No attack chains of depth >= {min_depth} found."

    lines = [f"Attack chains (depth >= {min_depth}, top {min(5, len(all_chains))}):"]
    for chain in all_chains[:5]:
        parts: list[str] = []
        for nid in chain:
            data = graph.nodes[nid]
            parts.append(f"[{nid}]({data.get('agent_id', '?')})")
        lines.append("  " + " -> ".join(parts))
    return "\n".join(lines)


def query_centrality_ranking(
    graph: nx.DiGraph, faction: str | None = None
) -> str:
    """Rank nodes by betweenness centrality."""
    if graph.number_of_nodes() == 0:
        return "No arguments in the graph yet."

    try:
        centrality = nx.betweenness_centrality(graph)
    except Exception:
        return "Could not compute centrality."

    items: list[tuple[float, str, str, str]] = []
    for nid, data in graph.nodes(data=True):
        if faction and not nid.startswith(faction):
            continue
        c = centrality.get(nid, 0.0)
        items.append((c, nid, data.get("agent_id", "?"), data.get("claim", "")[:80]))

    items.sort(key=lambda x: x[0], reverse=True)

    if not items:
        return "No nodes found matching criteria."

    lines = ["Nodes ranked by betweenness centrality (top 10):"]
    for c, nid, aid, claim in items[:10]:
        lines.append(f"- [{nid}] ({aid}) centrality={c:.3f}: \"{claim}\"")
    return "\n".join(lines)


def query_faction_balance(graph: nx.DiGraph) -> str:
    """Show argument count and acceptance per faction."""
    if graph.number_of_nodes() == 0:
        return "No arguments in the graph yet."

    faction_stats: dict[str, dict[str, int]] = {}
    for nid, data in graph.nodes(data=True):
        f = _faction_of(data.get("agent_id", ""))
        if f not in faction_stats:
            faction_stats[f] = {"total": 0, "attacked": 0, "defended": 0}
        faction_stats[f]["total"] += 1
        if graph.in_degree(nid) > 0:
            faction_stats[f]["attacked"] += 1
            own_faction = f
            defenders = [p for p in graph.predecessors(nid) if _faction_of(graph.nodes[p].get("agent_id", "")) == own_faction]
            if defenders:
                faction_stats[f]["defended"] += 1

    lines = ["Faction balance:"]
    for f, stats in sorted(faction_stats.items()):
        undefended = stats["attacked"] - stats["defended"]
        lines.append(
            f"- {f}: {stats['total']} arguments, "
            f"{stats['attacked']} attacked, "
            f"{stats['defended']} defended, "
            f"{undefended} undefended"
        )
    return "\n".join(lines)


def query_contradictions(graph: nx.DiGraph, faction: str | None = None) -> str:
    """Find mutual attack pairs (A attacks B and B attacks A) — dialectical cycles."""
    if graph.number_of_nodes() == 0:
        return "No arguments in the graph yet."

    cycles: list[tuple[str, str]] = []
    for u, v in graph.edges():
        if graph.has_edge(v, u) and (v, u) not in cycles and (u, v) not in cycles:
            if faction:
                if not (u.startswith(faction) or v.startswith(faction)):
                    continue
            cycles.append((u, v))

    if not cycles:
        return "No mutual attack cycles (contradictions) found."

    lines = [f"Mutual attack cycles ({len(cycles)}):"]
    for u, v in cycles[:10]:
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        lines.append(
            f"- [{u}] ({u_data.get('agent_id', '?')}) <-> [{v}] ({v_data.get('agent_id', '?')})"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Query registry and dispatcher
# ──────────────────────────────────────────────────────────────────────────

QUERY_CATALOG: dict[str, str] = {
    "undefended_attacks": (
        "undefended_attacks(faction) — Find attacks on your faction with no counter-attack. "
        "Params: faction (your faction prefix, e.g. 'GOV' or 'OPP')."
    ),
    "weak_opponent_nodes": (
        "weak_opponent_nodes(faction, min_centrality) — Find opponent nodes with low defense. "
        "Params: faction (opponent prefix), min_centrality (float, default 0.0)."
    ),
    "unattacked_nodes": (
        "unattacked_nodes(faction, sort_by) — Find unattacked opponent claims. "
        "Params: faction (opponent prefix), sort_by ('centrality')."
    ),
    "attack_chains": (
        "attack_chains(min_depth) — Find longest attack chains in the graph. "
        "Params: min_depth (int, default 2)."
    ),
    "centrality_ranking": (
        "centrality_ranking(faction) — Rank nodes by betweenness centrality. "
        "Params: faction (optional prefix to filter)."
    ),
    "faction_balance": (
        "faction_balance() — Show argument count and defense status per faction. No params."
    ),
    "contradictions": (
        "contradictions(faction) — Find mutual attack pairs (dialectical cycles). "
        "Params: faction (optional prefix to filter)."
    ),
}


def execute_query(
    graph: nx.DiGraph,
    query_name: str,
    params: dict[str, Any] | None = None,
) -> str:
    """Execute a named query against the argument graph.

    Returns a text summary of results. Fully deterministic.
    """
    params = params or {}

    if query_name == "undefended_attacks":
        return query_undefended_attacks(graph, faction=params.get("faction"))
    elif query_name == "weak_opponent_nodes":
        return query_weak_opponent_nodes(
            graph,
            faction=params.get("faction"),
            min_centrality=float(params.get("min_centrality", 0.0)),
        )
    elif query_name == "unattacked_nodes":
        return query_unattacked_nodes(
            graph,
            faction=params.get("faction"),
            sort_by=params.get("sort_by", "centrality"),
        )
    elif query_name == "attack_chains":
        return query_attack_chains(
            graph,
            min_depth=int(params.get("min_depth", 2)),
            faction=params.get("faction"),
        )
    elif query_name == "centrality_ranking":
        return query_centrality_ranking(graph, faction=params.get("faction"))
    elif query_name == "faction_balance":
        return query_faction_balance(graph)
    elif query_name == "contradictions":
        return query_contradictions(graph, faction=params.get("faction"))
    else:
        return f"Unknown query: {query_name}. Available: {', '.join(QUERY_CATALOG.keys())}"


def get_catalog_text() -> str:
    """Return the query catalog as prompt text for the LLM."""
    lines = ["Available SEARCH queries (graph-native, deterministic):"]
    for name, desc in QUERY_CATALOG.items():
        lines.append(f"  - {name}: {desc}")
    lines.append(
        "\nTo SEARCH, set action=\"SEARCH\" and put your query name in search_query "
        "and parameters in search_params (JSON object). Example:\n"
        '{"action": "SEARCH", "search_query": "undefended_attacks", '
        '"search_params": {"faction": "GOV"}, ...}'
    )
    return "\n".join(lines)
