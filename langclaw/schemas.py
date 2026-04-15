"""Pydantic schemas for structured LLM output and simulation logging.

These models enforce JSON structure on LLM responses and provide
typed records for the simulation event log.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AgentState(str, Enum):
    """Observable state of a LangClaw agent."""
    ACTIVE = "active"
    WORKING = "working"


class CognitivePhase(str, Enum):
    """Phase within the WORKING state's deliberative loop."""
    TRIAGE = "triage"
    THINK = "think"
    PLAN = "plan"
    EXECUTE = "execute"
    OBSERVE = "observe"


class Performative(str, Enum):
    """FIPA ACL-based communication acts for directed messaging."""
    REQUEST = "request"
    INFORM = "inform"
    PROPOSE = "propose"
    CONFIRM = "confirm"
    QUERY = "query"


class AgentAction(BaseModel):
    """Structured action returned by the LLM.

    The LLM must produce a JSON object matching this schema.
    - PASS    : the agent declines to participate.
    - DEBATE  : the agent introduces or attacks an argument in the graph.
    - SEARCH  : the agent retrieves a domain fact to improve future arguments.
    - READ    : the agent consumes a document from the shared knowledge pool.
    - MESSAGE : the agent sends a directed message to another agent.
    """

    action: Literal["PASS", "DEBATE", "SEARCH", "READ", "MESSAGE"] = Field(
        description="The action the agent takes this turn."
    )
    claim: str | None = Field(
        default=None,
        description="The argumentative claim (required when action=DEBATE).",
    )
    target_node_id: str | None = Field(
        default=None,
        description="ID of the node being attacked (None for a root argument).",
    )
    attack_type: Literal["undercut", "rebuttal"] | None = Field(
        default=None,
        description="Type of logical attack on the target node.",
    )
    send_to: str | None = Field(
        default=None,
        description="Target agent ID (required when action=MESSAGE).",
    )
    message_content: str | None = Field(
        default=None,
        description="Content of the directed message (required when action=MESSAGE).",
    )
    message_type: Literal["request", "inform", "propose", "confirm", "query"] | None = Field(
        default=None,
        description="FIPA ACL performative for the message.",
    )


class SimulationLog(BaseModel):
    """Single-tick log entry emitted by the simulation loop."""

    tick: int
    agent_id: str
    action: str
    claim: str | None = None
    target_node_id: str | None = None
    attack_type: str | None = None
    deficit_before: float
    deficit_after: float
    delta_phi: float = 0.0
    activation_prob: float = 0.0
    graph_nodes: int = 0
    graph_edges: int = 0
    consistency_rate: float = 0.0
    trigger: Literal["HOMEOSTATIC", "ROUTER", "FORCED"] = "FORCED"
    aaf_acceptance_ratio: float = 0.0
    aaf_defeat_cycles: int = 0
    aaf_dialectical_completeness: float = 0.0
    utility_debate: float = 0.0
    utility_search: float = 0.0
    utility_read: float = 0.0
    utility_pass: float = 0.0
    orchestration_mode: str = "hrrl"
    stimulus_event_id: str | None = None
    stimulus_utility: float = 0.0
    n_stimuli_evaluated: int = 0
    reward: float = 0.0
    q_values: dict[str, float] = Field(default_factory=dict)
    # Agent cognitive state
    agent_state: str = "active"
    cognitive_phase: str | None = None
    # Directed messaging
    send_to: str | None = None
    message_content: str | None = None
    message_type: str | None = None
    n_messages_received: int = 0
    # VSM subsystem
    vsm_system: str | None = None
