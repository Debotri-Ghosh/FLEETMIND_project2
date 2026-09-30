"""Bearing characteristic fault frequencies for the CWRU test rig.

A rolling-element bearing produces an impact every time a rolling element
passes the defect. The repetition rate depends only on bearing geometry and
shaft speed, so each defect location has its own characteristic frequency:

    BPFI  ball pass frequency, inner race
    BPFO  ball pass frequency, outer race
    BSF   ball spin frequency
    FTF   fundamental train (cage) frequency

These multipliers are the published values for the two bearings CWRU used,
and are what make the spectrum *interpretable*: energy concentrated near
BPFO is physical corroboration that an OuterRace call is real, rather than
the classifier keying on something incidental.

IMPORTANT, and stated plainly wherever this is surfaced: the CNN does **not**
use these frequencies. It consumes `ml/model.py:to_fft_magnitude()` (512-bin
log-scaled, z-normalized magnitude) and learns whatever separates the classes.
The characteristic frequencies here are independent, human-inspectable
evidence for a reviewer -- corroboration, not an explanation of the model's
internals. Do not present them as "what the model looked at".
"""

from __future__ import annotations

from dataclasses import dataclass

# Multipliers are in orders of shaft rotation (x shaft speed in Hz).
# Drive end: SKF 6205-2RS JEM.  Fan end: SKF 6203-2RS JEM.
BEARING_MULTIPLIERS: dict[str, dict[str, float]] = {
    "DE": {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 4.7135, "FTF": 0.3983},
    "FE": {"BPFI": 4.9469, "BPFO": 3.0530, "BSF": 3.9874, "FTF": 0.3817},
}

# Which characteristic frequency corroborates which predicted fault class.
# "Normal" has no defect frequency, and "Ball" is conventionally inspected at
# 2xBSF because a ball defect strikes both races once per revolution.
FAULT_TO_FREQUENCY: dict[str, str | None] = {
    "InnerRace": "BPFI",
    "OuterRace": "BPFO",
    "Ball": "BSF",
    "Normal": None,
}

BALL_USES_SECOND_HARMONIC = True


@dataclass
class CharacteristicFrequency:
    name: str
    hz: float
    band_low_hz: float
    band_high_hz: float
    harmonic: int = 1


def shaft_speed_hz(nominal_rpm: float) -> float:
    return nominal_rpm / 60.0


def characteristic_frequencies(nominal_rpm: float, sensor_position: str = "DE") -> dict[str, float]:
    """All four characteristic frequencies (Hz) for this shaft speed."""
    multipliers = BEARING_MULTIPLIERS.get(sensor_position.upper(), BEARING_MULTIPLIERS["DE"])
    speed = shaft_speed_hz(nominal_rpm)
    return {name: round(mult * speed, 2) for name, mult in multipliers.items()}


def matched_frequency_for_fault(
    fault_label: str,
    nominal_rpm: float,
    sensor_position: str = "DE",
    band_tolerance: float = 0.05,
) -> CharacteristicFrequency | None:
    """The characteristic frequency that corroborates this fault, with a band.

    Returns None for Normal (no defect frequency to look for) and for any
    label we have no mapping for -- callers must render "no band" rather than
    inventing one. The +/-5% default band absorbs the real slip and speed
    variation in the CWRU recordings; a zero-width line would look precise
    and be misleading.
    """
    base = fault_label.split(" ")[0] if fault_label else ""
    key = FAULT_TO_FREQUENCY.get(base)
    if key is None:
        return None

    freqs = characteristic_frequencies(nominal_rpm, sensor_position)
    hz = freqs[key]
    harmonic = 1
    if key == "BSF" and BALL_USES_SECOND_HARMONIC:
        hz *= 2
        harmonic = 2
    return CharacteristicFrequency(
        name=f"{harmonic}x{key}" if harmonic > 1 else key,
        hz=round(hz, 2),
        band_low_hz=round(hz * (1 - band_tolerance), 2),
        band_high_hz=round(hz * (1 + band_tolerance), 2),
        harmonic=harmonic,
    )


def spectrum(values: list[float], sampling_rate_hz: int, max_hz: float = 1000.0):
    """Plain magnitude spectrum in real Hz, for a human to look at.

    Deliberately NOT `to_fft_magnitude()`: that one is log-scaled and
    z-normalized for the network's benefit, which makes its y-axis
    uninterpretable to a person. This returns raw magnitudes against real
    frequencies so a reviewer can see energy sitting on a defect frequency.
    """
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [], []
    mags = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / sampling_rate_hz)
    keep = freqs <= max_hz
    return freqs[keep].round(2).tolist(), mags[keep].round(4).tolist()
