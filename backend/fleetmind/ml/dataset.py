"""Postgres-backed torch Dataset over the `signal_windows` table.

Each item is (waveform, label_index). The waveform is the stored
raw_values list (length = window_size, default 2048) reshaped to (1, N).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.db.models import SignalWindow

DEFAULT_WINDOW_SIZE = 2048


def _to_index(label: str) -> int:
    from .model import LABEL_MAP

    return LABEL_MAP.index(label)


class CWRUSignalDataset:
    """Thin dataset: pulls window rows from Postgres and converts to tensors.

    Implemented as a plain class (not torch.utils.data.Dataset) so the package
    stays importable without torch installed in lightweight contexts; the
    train/eval scripts wrap it with a tiny torch Dataset adapter.
    """

    def __init__(self, session: Session, window_size: int = DEFAULT_WINDOW_SIZE):
        self.session = session
        self.window_size = window_size
        self.rows = session.execute(
            select(SignalWindow.id, SignalWindow.raw_values, SignalWindow.label).order_by(
                SignalWindow.id
            )
        ).all()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        import torch

        row = self.rows[idx]
        values = [float(v) for v in (row.raw_values or [])]
        if len(values) < self.window_size:
            values += [0.0] * (self.window_size - len(values))
        else:
            values = values[: self.window_size]
        waveform = values if values else [0.0]
        x = torch.tensor(waveform, dtype=torch.float32).view(1, -1)
        y = _to_index(row.label)
        return x, y, row.id
