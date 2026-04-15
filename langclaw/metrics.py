"""Discourse quality metrics for LangClaw evaluation.

Implements established metrics from the argumentation and multi-agent literature
used to compare HRRL (endogenous) vs LangGraph (exogenous) orchestration.
"""

from __future__ import annotations

from langclaw.schemas import SimulationLog

# Stance lexicon for PRR text-based detection (English + Spanish)
_STANCE_LEXICON: frozenset[str] = frozenset({
    # English
    "agree", "disagree", "challenge", "support", "attack", "refute", "deny",
    "oppose", "endorse", "contradict", "rebut", "undercut",
    # Spanish
    "acuerdo", "desacuerdo", "contradice", "refuta", "apoya", "rechaza",
    "niega", "respalda", "contradigo", "rebate",
})


def peer_reference_rate(
    logs: list[SimulationLog],
    agent_ids: list[str],
) -> float:
    """PRR (text variant) — Marandi 2026, arXiv:2603.28813.

    Fraction of DEBATE turns in which the agent's claim contains:
      (a) an explicit mention of a peer agent's ID, AND
      (b) a stance word from the argumentation lexicon.

    PRR = 0 is the structural signature of independent parallel monologue.
    PRR = 1 means every debate turn explicitly engages with a peer's position.

    Parameters
    ----------
    logs:
        All SimulationLog entries for a simulation run.
    agent_ids:
        List of agent IDs participating in the debate.

    Returns
    -------
    float in [0, 1].  Returns 0.0 if no DEBATE turns with claims exist.
    """
    debate_turns = [l for l in logs if l.action == "DEBATE" and l.claim]
    if not debate_turns:
        return 0.0

    hits = 0
    for turn in debate_turns:
        text = turn.claim.lower()
        has_peer = any(aid.lower() in text for aid in agent_ids)
        has_stance = any(w in text for w in _STANCE_LEXICON)
        if has_peer and has_stance:
            hits += 1

    return hits / len(debate_turns)


def peer_reference_rate_graph(logs: list[SimulationLog]) -> float:
    """PRR (graph-structural variant).

    Fraction of DEBATE turns where the agent explicitly targets an existing
    node (target_node_id is not None).  In our structured debate format,
    a non-null target already implies a peer reference — the agent chose a
    specific argument to attack.

    This variant does not require text analysis and is deterministic given
    the simulation logs.  In well-connected debates the two PRR variants
    converge; divergence indicates arguments that mention peers without
    forming graph edges (or vice-versa).

    Returns 0.0 if no DEBATE turns exist.
    """
    debate_turns = [l for l in logs if l.action == "DEBATE"]
    if not debate_turns:
        return 0.0

    connected = sum(1 for t in debate_turns if t.target_node_id is not None)
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
