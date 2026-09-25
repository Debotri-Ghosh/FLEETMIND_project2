"""Tests for ml/predict.py's two modes.

Full mode (the documented offline batch workflow) wipes and reprocesses
everything, ignoring released_at. Incremental mode (used by the replay
pipeline) processes only windows that have "arrived" and are not already
predicted, and destroys nothing -- that is what makes repeated ticks safe.

A randomly-initialised BearingCNN is saved as the checkpoint: these tests are
about *which windows get processed*, not about prediction quality.
"""

from datetime import UTC, datetime

import pytest
import torch
from sqlalchemy.orm import sessionmaker

from fleetmind.db.models import (
    Depot,
    Experiment,
    ModelPrediction,
    SignalWindow,
    seed_depots_payload,
)
from fleetmind.ml.model import LABEL_MAP, BearingCNN
from fleetmind.ml.predict import predict_depot


@pytest.fixture()
def checkpoint(tmp_path):
    path = tmp_path / "cnn_state_dict.pt"
    torch.save(BearingCNN(num_classes=len(LABEL_MAP)).state_dict(), path)
    return path


@pytest.fixture()
def windows_session(db_engine):
    """Depot 1 with one experiment and six windows, all unreleased."""
    session = sessionmaker(bind=db_engine, expire_on_commit=False)()
    for row in seed_depots_payload():
        session.add(Depot(**row))
    session.flush()
    exp = Experiment(
        depot_id=1,
        fault_type="InnerRace",
        fault_diameter_in=0.007,
        sensor_position="DE",
        sampling_rate_hz=12000,
        source_file="12k_DE/Load_0/InnerRace/IR007_0.mat",
        source_variable="X097_DE_time",
        n_samples=4096,
    )
    session.add(exp)
    session.flush()
    for i in range(6):
        session.add(
            SignalWindow(
                experiment_id=exp.id,
                window_index=i,
                start_sample=i * 2048,
                end_sample=(i + 1) * 2048,
                raw_values=[0.01 * i] * 2048,
                label="InnerRace",
                released_at=None,
            )
        )
    session.commit()
    yield session
    session.close()


def _predicted_window_ids(session):
    return sorted(session.query(ModelPrediction.window_id).all())


def test_incremental_skips_unreleased_windows(windows_session, checkpoint):
    """Nothing has arrived yet, so an incremental run must do nothing at all."""
    n = predict_depot(1, windows_session, checkpoint, incremental=True)
    windows_session.commit()

    assert n == 0
    assert windows_session.query(ModelPrediction).count() == 0


def test_incremental_processes_only_released_windows(windows_session, checkpoint):
    released = windows_session.query(SignalWindow).order_by(SignalWindow.id).limit(2).all()
    for w in released:
        w.released_at = datetime.now(UTC)
    windows_session.commit()

    n = predict_depot(1, windows_session, checkpoint, incremental=True)
    windows_session.commit()

    assert n == 2
    assert _predicted_window_ids(windows_session) == sorted((w.id,) for w in released)


def test_incremental_is_repeatable_without_duplicating(windows_session, checkpoint):
    """The replay worker calls this on every tick -- a tick with no new
    arrivals must be a no-op, not a duplicate-insert or a wipe."""
    for w in windows_session.query(SignalWindow).order_by(SignalWindow.id).limit(3).all():
        w.released_at = datetime.now(UTC)
    windows_session.commit()

    first = predict_depot(1, windows_session, checkpoint, incremental=True)
    windows_session.commit()
    second = predict_depot(1, windows_session, checkpoint, incremental=True)
    windows_session.commit()

    assert (first, second) == (3, 0)
    assert windows_session.query(ModelPrediction).count() == 3


def test_incremental_picks_up_newly_released_windows(windows_session, checkpoint):
    """The core replay behaviour: release a batch, process, release more,
    process again -- each tick only touches what newly arrived."""
    all_windows = windows_session.query(SignalWindow).order_by(SignalWindow.id).all()
    for w in all_windows[:2]:
        w.released_at = datetime.now(UTC)
    windows_session.commit()
    predict_depot(1, windows_session, checkpoint, incremental=True)
    windows_session.commit()

    for w in all_windows[2:5]:
        w.released_at = datetime.now(UTC)
    windows_session.commit()
    n = predict_depot(1, windows_session, checkpoint, incremental=True)
    windows_session.commit()

    assert n == 3
    assert windows_session.query(ModelPrediction).count() == 5


def test_full_mode_ignores_released_at_and_reprocesses_everything(windows_session, checkpoint):
    """The documented offline workflow is unchanged by this feature: it still
    processes every window regardless of arrival state."""
    n = predict_depot(1, windows_session, checkpoint, incremental=False)
    windows_session.commit()

    assert n == 6
    assert windows_session.query(ModelPrediction).count() == 6


def test_full_mode_wipes_and_rebuilds_predictions(windows_session, checkpoint):
    """A stale prediction from an older model version must not survive a full
    re-run. (Asserting on row ids would only test sqlite's rowid reuse, so use
    a sentinel model_version that could only exist if the wipe didn't happen.)"""
    window_id = windows_session.query(SignalWindow.id).order_by(SignalWindow.id).first()[0]
    windows_session.add(
        ModelPrediction(
            window_id=window_id,
            model_version="STALE-should-be-wiped",
            predicted_label="Normal",
            confidence=0.1,
        )
    )
    windows_session.commit()

    predict_depot(1, windows_session, checkpoint, incremental=False)
    windows_session.commit()

    versions = {row.model_version for row in windows_session.query(ModelPrediction).all()}
    assert "STALE-should-be-wiped" not in versions
    assert windows_session.query(ModelPrediction).count() == 6
