"""
Event dataclasses for LangClaw's event-driven architecture.

Agents receive these events via asyncio queues and react autonomously
based on their internal homeostatic state — no external orchestrator needed.

Event types
-----------
TickElapsedEvent     : emitted by the environment at each simulation tick.
NewArgumentEvent     : emitted when any agent adds a public argument to the graph.
DirectMessageEvent   : directed agent-to-agent communication (FIPA ACL performatives).
SimulationEndEvent   : shutdown sentinel.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class TickElapsedEvent:
    """Emitted by the environment at each simulation tick.
    Triggers drive decay in all agents.
    """
    tick: int


@dataclass(frozen=True)
class NewArgumentEvent:
    """Emitted when any agent successfully adds an argument to the graph.
    Other agents observe this and update their working memory.

    faction / targets_faction enable stimulus evaluation without
    re-querying the graph: the agent can immediately assess whether
    this event is relevant to its own faction.
    """
    tick: int
    node_id: str
    agent_id: str
    claim: str
    delta_phi: float
    attack_type: str | None = None
    target_node_id: str | None = None
    faction: str = ""
    targets_faction: str | None = None


@dataclass(frozen=True)
class DirectMessageEvent:
    """Directed agent-to-agent communication.

    Routed ONLY to the target agent's queue (not broadcast).
    Performatives follow FIPA ACL conventions:
      request  — asks another agent to perform a specific action
      inform   — provides information to another agent
      propose  — suggests a course of action
      confirm  — verifies a piece of information
      query    — requests information or clarification
    """
    tick: int
    from_agent: str
    to_agent: str
    content: str
    performative: str  # request | inform | propose | confirm | query


@dataclass(frozen=True)
class SimulationEndEvent:
    """Emitted once by the environment to signal all agents to shut down."""
    pass


Event = Union[
    TickElapsedEvent,
    NewArgumentEvent,
    DirectMessageEvent,
    SimulationEndEvent,
]
