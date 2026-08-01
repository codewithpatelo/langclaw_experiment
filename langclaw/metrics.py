"""Discourse quality metrics for LangClaw evaluation.

Implements established metrics from the argumentation and multi-agent literature
used to compare HRRL (endogenous) vs LangGraph (exogenous) orchestration.
"""

from __future__ import annotations

from langclaw.schemas import SimulationLog


def peer_reference_rate(
    logs: list[SimulationLog],
    agent_ids: list[str],
) -> float:
    """PRR (text variant) — content engagement metric.

    Fraction of DEBATE turns in which the agent's claim shares significant
    token overlap with the claim it attacks (target_node_id).  This detects
    whether the agent actually addresses the content of the opposing argument,
    rather than producing disconnected monologue.

    A turn counts as a peer reference if:
      (a) it has a valid target_node_id pointing to a real node, AND
      (b) the token overlap between the agent's claim and the target's
          claim exceeds a threshold (currently 0.15).

    PRR = 0 is the structural signature of independent parallel monologue.
    PRR = 1 means every debate turn engages with the content of a peer's
    argument.

    Parameters
    ----------
    logs:
        All SimulationLog entries for a simulation run.
    agent_ids:
        List of agent IDs participating in the debate (retained for
        API compatibility; not used in the content-based variant).

    Returns
    -------
    float in [0, 1].  Returns 0.0 if no DEBATE turns with claims exist.
    """
    from langclaw.delp_graph import _token_overlap

    debate_turns = [l for l in logs if l.action == "DEBATE" and l.claim]
    if not debate_turns:
        return 0.0

    # Build a lookup: node_id -> claim text
    node_claims: dict[str, str] = {}
    for l in logs:
        if l.node_id and l.claim:
            node_claims[l.node_id] = l.claim

    ENGAGEMENT_THRESHOLD = 0.15

    hits = 0
    for turn in debate_turns:
        if not turn.target_node_id:
            continue
        target_claim = node_claims.get(turn.target_node_id)
        if not target_claim:
            continue
        overlap = _token_overlap(turn.claim, target_claim)
        if overlap >= ENGAGEMENT_THRESHOLD:
            hits += 1

    return hits / len(debate_turns)


def peer_reference_rate_graph(logs: list[SimulationLog]) -> float:
    """PRR (graph-structural variant).

    Fraction of DEBATE turns where the agent targets a node that actually
    exists in the graph (target_node_id matches a real node_id from the
    same run).  A non-null target that doesn't correspond to any real node
    indicates an LLM hallucination or a connectivity bug, and should not
    count as a peer reference.

    This variant does not require text analysis and is deterministic given
    the simulation logs.  In well-connected debates the two PRR variants
    converge; divergence indicates arguments that mention peers without
    forming graph edges (or vice-versa).

    Returns 0.0 if no DEBATE turns exist.
    """
    debate_turns = [l for l in logs if l.action == "DEBATE"]
    if not debate_turns:
        return 0.0

    real_node_ids = {l.node_id for l in logs if l.node_id}
    connected = sum(
        1 for t in debate_turns
        if t.target_node_id is not None and t.target_node_id in real_node_ids
    )
    return connected / len(debate_turns)


def initiative_ratio(logs: list[SimulationLog]) -> float:
    """IR (Initiative Ratio) — validity check, NOT a comparison metric.

    IR = count(trigger == "HOMEOSTATIC") / count(action != "PASS")

    Measures the fraction of active turns that were self-initiated by the
    agent's homeostatic sigmoid.  Expected values:
      - HRRL mode: IR ≈ 1.0 (all active turns are self-initiated)
      - LangGraph mode: IR ≈ 0.0 (all active turns are externally routed)

    This is a structural property of the orchestration design, not a
    discourse quality outcome.  It is reported as a validity check only.

    Returns 0.0 if no active turns exist.
    """
    active_turns = [l for l in logs if l.action != "PASS"]
    if not active_turns:
        return 0.0

    homeostatic = sum(1 for t in active_turns if t.trigger == "HOMEOSTATIC")
    return homeostatic / len(active_turns)
