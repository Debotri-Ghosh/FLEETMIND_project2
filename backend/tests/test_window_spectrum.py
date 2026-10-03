"""Tests for GET /signal-window/{id}/spectrum -- the Frequency Lab's data.

The endpoint exists to let the UI show that the frequency view is *derived*
from the waveform already on screen, so these assert the two halves describe
the same samples and that the spectrum is in real Hz rather than bin indices.
"""

import math

from fleetmind.db.models import Experiment, SignalWindow


def _window_id(session) -> int:
    return session.query(SignalWindow).first().id


def test_spectrum_returns_both_domains_for_the_same_window(client, seeded_session):
    wid = _window_id(seeded_session)

    body = client.get(f"/signal-window/{wid}/spectrum").json()

    assert body["window_id"] == wid
    assert body["label"] == "InnerRace"
    assert body["sampling_rate_hz"] == 12000
    # the waveform is the stored samples, unmodified
    assert body["raw_values"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert body["n_samples"] == 8
    # and the spectrum is computed from them, not fetched from anywhere
    assert len(body["spectrum_freqs_hz"]) == len(body["spectrum_magnitudes"])
    assert len(body["spectrum_freqs_hz"]) > 0


def test_spectrum_x_axis_is_real_hz_not_bin_indices(client, seeded_session, db_engine):
    """A synthetic 1500 Hz tone must peak at 1500 Hz.

    This is the check that catches the classic FFT bug of plotting bin number
    and calling it frequency.
    """
    from sqlalchemy.orm import sessionmaker

    session = sessionmaker(bind=db_engine, expire_on_commit=False)()
    experiment = session.query(Experiment).first()
    rate, tone, n = 12000, 1500.0, 512
    window = SignalWindow(
        experiment_id=experiment.id,
        window_index=99,
        start_sample=0,
        end_sample=n,
        raw_values=[math.sin(2 * math.pi * tone * i / rate) for i in range(n)],
        label="Normal",
    )
    session.add(window)
    session.commit()

    body = client.get(f"/signal-window/{window.id}/spectrum").json()

    peak = max(
        zip(body["spectrum_freqs_hz"], body["spectrum_magnitudes"], strict=True),
        key=lambda p: p[1],
    )[0]
    assert abs(peak - tone) < rate / n, f"peak at {peak} Hz, expected ~{tone} Hz"
    session.close()


def test_spectrum_marks_the_defect_band_for_the_documented_fault(client, seeded_session):
    """The fixture window is InnerRace, so the band must be BPFI."""
    wid = _window_id(seeded_session)

    body = client.get(f"/signal-window/{wid}/spectrum").json()

    matched = body["matched_frequency"]
    assert matched is not None
    assert matched["name"] == "BPFI"
    assert matched["band_low_hz"] < matched["hz"] < matched["band_high_hz"]
    assert set(body["all_characteristic_frequencies"]) == {"BPFI", "BPFO", "BSF", "FTF"}


def test_band_energy_fraction_is_a_real_share(client, seeded_session):
    wid = _window_id(seeded_session)

    body = client.get(f"/signal-window/{wid}/spectrum").json()

    assert 0.0 <= body["band_energy_fraction"] <= 1.0
    assert body["overall_peak_hz"] is not None


def test_a_normal_window_gets_no_defect_band(client, seeded_session, db_engine):
    """Normal has no characteristic fault frequency, so the UI must draw none."""
    from sqlalchemy.orm import sessionmaker

    session = sessionmaker(bind=db_engine, expire_on_commit=False)()
    experiment = session.query(Experiment).first()
    window = SignalWindow(
        experiment_id=experiment.id,
        window_index=98,
        start_sample=0,
        end_sample=8,
        raw_values=[0.01] * 8,
        label="Normal",
    )
    session.add(window)
    session.commit()

    body = client.get(f"/signal-window/{window.id}/spectrum").json()

    assert body["matched_frequency"] is None
    assert body["band_energy_fraction"] is None
    session.close()


def test_unknown_window_is_404(client):
    assert client.get("/signal-window/999999/spectrum").status_code == 404


def test_spectrum_requires_authentication(anon_client, seeded_session):
    wid = _window_id(seeded_session)
    assert anon_client.get(f"/signal-window/{wid}/spectrum").status_code == 401
