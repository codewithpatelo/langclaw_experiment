"""
Event dataclasses for LangClaw's event-driven architecture.

Agents receive these events via asyncio queues and react autonomously
based on their internal homeostatic state — no external orchestrator needed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class TickElapsedEvent:
    """
    Emitted by the environment at each simulation tick.
    Triggers drive decay in all agents.
    """
    tick: int


@dataclass(frozen=True)
class NewArgumentEvent:
    """
    Emitted when any agent successfully adds an argument to the graph.
    Other agents observe this and update their working memory.
    """
    tick: int
    node_id: str
    agent_id: str
    claim: str
    delta_phi: float
    attack_type: str | None = None
    target_node_id: str | None = None


@dataclass(frozen=True)
class SimulationEndEvent:
    """
    Emitted once by the environment to signal all agents to shut down.
    """
    pass


# Union type for type-checking event queues
Event = Union[TickElapsedEvent, NewArgumentEvent, SimulationEndEvent]
