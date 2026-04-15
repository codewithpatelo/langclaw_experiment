"""LangGraph-compiled cognitive loop for exogenous orchestration.

Materializes the cognitive loop (THINK → PLAN → EXECUTE → OBSERVE) as a
LangGraph StateGraph.  The graph's conditional edges control the flow —
NOT the agent's internal sigmoid gate, drive function, or Q-learner.

Key difference from HRRL:
  - HRRL: endogenous regulation (deficit → drive D(δ)=(δ-ε)² → sigmoid →
    Q-learner selects action → reward = drive reduction → TD update).
  - LangGraph: exogenous orchestration (router selects agent → compiled graph
    structures cognitive phases → LLM decides action openly).

Same agents, same action space, same memory, same states.
Different control locus: endogenous vs exogenous.

Variables that belong ONLY to HRRL (never used here):
  drive, activation_prob, sigmoid gate, Q-learner, homeostatic reward.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)


class CognitiveState(TypedDict):
    """State flowing through the cognitive loop graph.

    ``deficit`` is an observable agent property (epistemic need), present as
    passive information.  It is NOT used in any conditional edge — the graph
    decides flow based on discourse context and budget, not homeostatic
    variables.
    """
    agent_id: str
    tick: int
    deficit: float
    graph_context: str
    target_ids: list[str]
    faction_agents: str
    memory_context: str
    messages_context: str
    stimulus_context: str
    role_prompt: str
    # Outputs
    action: str
    claim: str | None
    target_node_id: str | None
    attack_type: str | None
    send_to: str | None
    message_content: str | None
    message_type: str | None
    delta_phi: float
    phase: str
    should_act: bool
    llm_response: str | None
    budget_ok: bool


def build_cognitive_graph() -> Any:
    """Build and compile the cognitive loop as a LangGraph StateGraph.

    Returns a compiled graph that can be invoked with CognitiveState.
    The graph implements: THINK → (PASS | PLAN → EXECUTE → OBSERVE)

    The decision to act or pass is made by the graph's conditional edges,
    not by any internal agent mechanism.
    """
    graph = StateGraph(CognitiveState)

    graph.add_node("think", _think_node)
    graph.add_node("plan", _plan_node)
    graph.add_node("execute", _execute_node)
    graph.add_node("observe", _observe_node)
    graph.add_node("pass_node", _pass_node)

    graph.add_edge(START, "think")

    graph.add_conditional_edges(
        "think",
        _should_act_decision,
        {"plan": "plan", "pass": "pass_node"},
    )
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "observe")
    graph.add_edge("observe", END)
    graph.add_edge("pass_node", END)

    return graph.compile()


# ──────────────────────────────────────────────────────────────────────────────
# Graph nodes
# ──────────────────────────────────────────────────────────────────────────────


def _think_node(state: CognitiveState) -> dict:
    """Assess the situation based on discourse context and resources.

    The router already selected this agent — the think node validates that
    execution is feasible (budget) and that there is discourse context,
    messages, or stimulus worth responding to.

    NO homeostatic variables (drive, sigmoid, deficit threshold) are used
    here.  Those belong exclusively to HRRL's endogenous regulation.
    """
    has_context = bool(
        state["graph_context"]
        and state["graph_context"]
        != "No hay argumentos aun. Presenta tu posicion inicial."
    )
    has_messages = state["messages_context"] != "No messages."
    has_stimulus = (
        state["stimulus_context"]
        != "No specific stimulus. Act proactively."
    )
    has_budget = state["budget_ok"]

    should_act = has_budget and (has_context or has_messages or has_stimulus)

    return {
        "phase": "think",
        "should_act": should_act,
    }


def _should_act_decision(state: CognitiveState) -> Literal["plan", "pass"]:
    """Conditional edge: proceed to PLAN or PASS.

    In HRRL the sigmoid gate decides probabilistically based on drive.
    Here the graph decides deterministically based on discourse state — the
    router already chose this agent, so we only gate on budget and context.
    """
    if not state["budget_ok"]:
        return "pass"
    if not state["should_act"]:
        return "pass"
    return "plan"


def _plan_node(state: CognitiveState) -> dict:
    """Plan phase: the graph determines the action will be executed."""
    return {"phase": "plan"}


def _execute_node(state: CognitiveState) -> dict:
    """Execute phase: marker node. Actual LLM call happens in the simulation
    layer after the graph returns, since LLM calls require the full agent
    infrastructure (client, graph object, budget).

    The graph sets phase='execute' to signal the simulation to call the LLM.
    """
    return {"phase": "execute", "action": "PENDING_LLM"}


def _pass_node(state: CognitiveState) -> dict:
    """Agent passes this tick."""
    return {
        "phase": "pass",
        "action": "PASS",
        "should_act": False,
    }


def _observe_node(state: CognitiveState) -> dict:
    """Observe phase: marker for logging. Actual reward computation
    happens in the simulation layer.
    """
    return {"phase": "observe"}
