"""FleetMind multi-agent diagnosis/risk/maintenance system.

Four agents orchestrated as a LangGraph state machine:
    monitor     watches incoming predictions, flags anomalies
    diagnostic  confirms/refines the fault diagnosis from signal features
    risk        computes a risk score from fault type + depot load
    maintenance recommends an action from diagnosis + risk

Each agent writes its output to the shared AgentState; the maintenance
agent persists the final decision to the `agent_decisions` table.
"""

from fleetmind.agents.graph import build_agent_graph
from fleetmind.agents.run import run_agent_pipeline
from fleetmind.agents.state import AgentState

__all__ = ["AgentState", "build_agent_graph", "run_agent_pipeline"]
