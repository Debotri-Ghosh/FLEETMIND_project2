"""Signal-window endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import (
    MatchedFrequency,
    WindowDetail,
    WindowSpectrum,
    WindowSummary,
)
from fleetmind.auth.deps import require_any_role
from fleetmind.db import session as db_session
from fleetmind.db.models import Depot, Experiment, SignalWindow
from fleetmind.ml.bearing_freq import (
    characteristic_frequencies,
    matched_frequency_for_fault,
    spectrum,
)

# Any signed-in role may read the research/ML layer; nobody anonymous may.
router = APIRouter(tags=["windows"], dependencies=[Depends(require_any_role)])


@router.get("/experiments/{experiment_id}/windows", response_model=list[WindowSummary])
def list_windows_for_experiment(
    experiment_id: int,
    limit: int = Query(default=10, le=100),
    db: Session = Depends(db_session.get_session),
):
    query = (
        select(
            SignalWindow.id,
            SignalWindow.experiment_id,
            SignalWindow.window_index,
            SignalWindow.start_sample,
            SignalWindow.end_sample,
            SignalWindow.label,
        )
        .where(SignalWindow.experiment_id == experiment_id)
        .order_by(SignalWindow.window_index)
        .limit(limit)
    )
    return [WindowSummary(**row._mapping) for row in db.execute(query)]


@router.get("/signal-window/{window_id}", response_model=WindowDetail)
def get_signal_window(window_id: int, db: Session = Depends(db_session.get_session)):
    window = db.get(SignalWindow, window_id)
    if window is None:
        raise HTTPException(status_code=404, detail="window not found")
    return WindowDetail(
        id=window.id,
        experiment_id=window.experiment_id,
        window_index=window.window_index,
        start_sample=window.start_sample,
        end_sample=window.end_sample,
        label=window.label,
        raw_values=window.raw_values,
    )


@router.get("/signal-window/{window_id}/spectrum", response_model=WindowSpectrum)
def get_window_spectrum(window_id: int, db: Session = Depends(db_session.get_session)):
    """The same window, in both domains.

    Returns the stored samples *and* their Fourier transform in one response
    so the UI can show a reviewer that the frequency view is derived from the
    waveform already on screen -- the same recording, re-expressed -- rather
    than being a second, separately collected measurement.

    The defect band is keyed to the window's **documented** fault class, since
    this view is about the physics of the recording rather than about what the
    model said. Those frequencies are bearing geometry; the CNN never sees them.

    The spectrum runs to Nyquist rather than being cropped, so where the energy
    actually sits is visible even when that is nowhere near the defect band.
    """
    row = db.execute(
        select(SignalWindow, Experiment, Depot)
        .join(Experiment, Experiment.id == SignalWindow.experiment_id)
        .join(Depot, Depot.id == Experiment.depot_id)
        .where(SignalWindow.id == window_id)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="window not found")

    window, experiment, depot = row
    values = [float(v) for v in (window.raw_values or [])]

    # The FULL spectrum to Nyquist, deliberately -- not the 1 kHz crop the
    # evidence panel uses. On real CWRU windows the dominant peak sits above
    # 1 kHz, so cropping would hide exactly the thing a reviewer should see and
    # would make "overall peak" a statement about the crop rather than the
    # signal. Both the energy fraction and the peak below are computed over
    # this full range.
    freqs, mags = spectrum(values, experiment.sampling_rate_hz, max_hz=float("inf"))
    matched = matched_frequency_for_fault(
        window.label, depot.nominal_rpm or 1797, experiment.sensor_position
    )

    band_fraction = None
    if matched is not None and mags:
        total = sum(m * m for m in mags)
        in_band = sum(
            m * m
            for f, m in zip(freqs, mags, strict=True)
            if matched.band_low_hz <= f <= matched.band_high_hz
        )
        band_fraction = round(in_band / total, 6) if total else 0.0

    peak_hz = max(zip(freqs, mags, strict=True), key=lambda p: p[1])[0] if mags else None

    return WindowSpectrum(
        window_id=window.id,
        experiment_id=experiment.id,
        label=window.label,
        sampling_rate_hz=experiment.sampling_rate_hz,
        n_samples=len(values),
        duration_seconds=round(len(values) / experiment.sampling_rate_hz, 5)
        if experiment.sampling_rate_hz
        else 0.0,
        raw_values=values,
        spectrum_freqs_hz=freqs,
        spectrum_magnitudes=mags,
        matched_frequency=MatchedFrequency(**vars(matched)) if matched else None,
        all_characteristic_frequencies=characteristic_frequencies(
            depot.nominal_rpm or 1797, experiment.sensor_position
        ),
        band_energy_fraction=band_fraction,
        overall_peak_hz=peak_hz,
    )
