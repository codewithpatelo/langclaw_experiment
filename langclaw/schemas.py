"""Pydantic schemas for structured LLM output and simulation logging.

These models enforce JSON structure on LLM responses and provide
typed records for the simulation event log.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    """Structured action returned by the LLM.

    The LLM must produce a JSON object matching this schema.
    - PASS   : the agent declines to participate this tick.
    - DEBATE : the agent introduces or attacks an argument in the graph.
    - SEARCH : the agent retrieves a domain fact to improve future arguments.
    - READ   : the agent consumes a document from the shared knowledge pool.
    """

    action: Literal["PASS", "DEBATE", "SEARCH", "READ"] = Field(
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
    tau_bench: float = 0.0
    # Utility scores that drove the action selection (HRRL mode only)
    utility_debate: float = 0.0
    utility_search: float = 0.0
    utility_read: float = 0.0
    utility_pass: float = 0.0
    # Orchestration mode used for this run
    orchestration_mode: str = "hrrl"
