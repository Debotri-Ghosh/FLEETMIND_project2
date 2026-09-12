"""LangGraph orchestration for the FleetMind multi-agent pipeline.

The graph is a simple linear chain:

    monitor -> diagnostic -> risk -> maintenance

Each node is one of the pure agent functions in agents.py. The graph
compiles into a callable that takes an AgentState and returns the
fully-populated AgentState after all four agents have run.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from fleetmind.agents.agents import (
    diagnostic_agent,
    maintenance_agent,
    monitor_agent,
    risk_agent,
)
from fleetmind.agents.state import AgentState


def build_agent_graph():
    """Build and compile the LangGraph agent pipeline."""
    graph = StateGraph(AgentState)

    graph.add_node("monitor", monitor_agent)
    graph.add_node("diagnostic", diagnostic_agent)
    graph.add_node("risk", risk_agent)
    graph.add_node("maintenance", maintenance_agent)

    graph.set_entry_point("monitor")
    graph.add_edge("monitor", "diagnostic")
    graph.add_edge("diagnostic", "risk")
    graph.add_edge("risk", "maintenance")
    graph.add_edge("maintenance", END)

    return graph.compile()
