"""Replay engine: simulates a live sensor feed over the static CWRU archive.

CWRU is a fixed historical dataset — there is no real stream to react to, so
putting a scheduler on it as-is would just recompute identical results on a
timer. Instead, this package progressively *releases* already-ingested signal
windows (``signal_windows.released_at``) as though they were arriving from
sensors, and the pipeline processes each newly-arrived batch incrementally.

This is trace-driven replay of archived data, and the UI labels it as such.
It is never presented as live telemetry.
"""

from fleetmind.replay.engine import (
    ReplayStatus,
    release_next_batch,
    replay_status,
    reset_replay,
)

__all__ = [
    "ReplayStatus",
    "release_next_batch",
    "replay_status",
    "reset_replay",
]
