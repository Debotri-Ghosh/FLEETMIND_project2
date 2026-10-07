"""Calibrated probabilistic decision layer.

This package is replacing the hand-tuned risk heuristic in `agents/agents.py`
as FleetMind's authoritative decision path. The governing rule is that no number
influencing a maintenance action may be invented: it is either fitted on
held-out data, or an explicit domain input carrying its units and a sensitivity
analysis.

Phase 0 (this code) builds no new intelligence. It measures the baseline, so a
later improvement can be claimed against a real before-number rather than
against an impression. See
`docs/plans/2026-10-06-decision-intelligence/PLAN.md`.
"""
