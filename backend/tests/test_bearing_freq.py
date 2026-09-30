"""Tests for bearing characteristic-frequency maths.

These are checked against the published SKF 6205-2RS JEM / 6203-2RS JEM
multipliers CWRU documents for its rig, so a wrong constant fails here rather
than silently drawing a misleading band on an evidence plot.
"""

import pytest

from fleetmind.ml.bearing_freq import (
    characteristic_frequencies,
    matched_frequency_for_fault,
    shaft_speed_hz,
    spectrum,
)


def test_shaft_speed_converts_rpm_to_hz():
    assert shaft_speed_hz(1797) == pytest.approx(29.95, abs=0.01)


def test_drive_end_frequencies_match_published_6205_values():
    """At 1797 rpm (CWRU's 0 HP case) the 6205-2RS JEM frequencies are the
    textbook ~162 Hz BPFI / ~107 Hz BPFO."""
    f = characteristic_frequencies(1797, "DE")
    assert f["BPFI"] == pytest.approx(162.19, abs=0.5)
    assert f["BPFO"] == pytest.approx(107.36, abs=0.5)
    assert f["BSF"] == pytest.approx(141.17, abs=0.5)
    assert f["FTF"] == pytest.approx(11.93, abs=0.1)


def test_fan_end_bearing_has_different_frequencies():
    de = characteristic_frequencies(1797, "DE")
    fe = characteristic_frequencies(1797, "FE")
    assert de["BPFI"] != fe["BPFI"]
    assert fe["BPFI"] == pytest.approx(148.16, abs=0.5)


def test_unknown_sensor_position_falls_back_to_drive_end():
    assert characteristic_frequencies(1797, "XX") == characteristic_frequencies(1797, "DE")


def test_frequencies_scale_with_shaft_speed():
    """Lower rpm must move every defect frequency down, or the band would be
    drawn in the wrong place for the 3 HP site."""
    slow = characteristic_frequencies(1730, "DE")
    fast = characteristic_frequencies(1797, "DE")
    assert slow["BPFO"] < fast["BPFO"]


def test_inner_race_matches_bpfi_with_a_band():
    m = matched_frequency_for_fault("InnerRace", 1797, "DE")
    assert m is not None
    assert m.name == "BPFI"
    assert m.hz == pytest.approx(162.19, abs=0.5)
    assert m.band_low_hz < m.hz < m.band_high_hz
    assert m.harmonic == 1


def test_outer_race_matches_bpfo():
    m = matched_frequency_for_fault("OuterRace", 1797, "DE")
    assert m is not None and m.name == "BPFO"


def test_ball_fault_uses_the_second_harmonic():
    """A ball defect strikes both races per revolution, so 2xBSF is the
    conventional inspection frequency."""
    m = matched_frequency_for_fault("Ball", 1797, "DE")
    assert m is not None
    assert m.harmonic == 2
    assert m.name == "2xBSF"
    assert m.hz == pytest.approx(2 * 141.17, abs=1.0)


def test_normal_has_no_matched_frequency():
    """There is no defect frequency for a healthy bearing -- the UI must show
    no band rather than inventing one."""
    assert matched_frequency_for_fault("Normal", 1797, "DE") is None


def test_unconfirmed_suffix_still_matches_base_label():
    """diagnostic_agent can return 'InnerRace (unconfirmed - low signal
    energy)'; the band should still resolve off the base class."""
    m = matched_frequency_for_fault("InnerRace (unconfirmed", 1797, "DE")
    assert m is not None and m.name == "BPFI"


def test_unknown_label_yields_no_band():
    assert matched_frequency_for_fault("Gearbox", 1797, "DE") is None


def test_band_width_follows_tolerance():
    m = matched_frequency_for_fault("OuterRace", 1797, "DE", band_tolerance=0.10)
    assert m is not None
    assert m.band_low_hz == pytest.approx(m.hz * 0.90, abs=0.1)
    assert m.band_high_hz == pytest.approx(m.hz * 1.10, abs=0.1)


# --- spectrum -----------------------------------------------------------


def test_spectrum_puts_a_known_tone_in_the_right_bin():
    """A synthetic 120 Hz sine must peak at 120 Hz -- this is what proves the
    frequency axis is real Hz and not bin indices."""
    import math

    sr = 12000
    n = 2048
    values = [math.sin(2 * math.pi * 120.0 * (i / sr)) for i in range(n)]

    freqs, mags = spectrum(values, sr)

    peak_hz = freqs[mags.index(max(mags))]
    assert peak_hz == pytest.approx(120.0, abs=6.0)  # bin spacing ~5.86 Hz


def test_spectrum_is_truncated_to_the_requested_range():
    freqs, mags = spectrum([0.1] * 2048, 12000, max_hz=500.0)
    assert len(freqs) == len(mags)
    assert max(freqs) <= 500.0


def test_spectrum_of_empty_window_is_empty_not_an_error():
    assert spectrum([], 12000) == ([], [])
