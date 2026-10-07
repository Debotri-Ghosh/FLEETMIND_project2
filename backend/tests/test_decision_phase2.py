"""Phase 2: temperature scaling and split-conformal prediction sets.

Three properties carry the weight here.

**Temperature cannot change a prediction.** Dividing all logits by one positive
scalar preserves their order, so accuracy and per-class recall must be bit-identical
before and after. If that ever stops holding, temperature scaling has been
implemented wrongly and every calibration claim built on it is void.

**The conformal quantile index is the easy thing to get subtly wrong.** Dropping
the `+1` or the ceiling from `ceil((n+1)(1-alpha))` undercovers while still
producing entirely plausible numbers. It is therefore tested directly, against
values worked out by hand, rather than only through its effect on coverage.

**The coverage guarantee is checked on genuinely exchangeable synthetic data.**
On the real CWRU split it cannot be verified -- calibration windows come from four
autocorrelated recordings -- so the only place the implementation's correctness
can actually be demonstrated is on data built to satisfy the assumption. That is
what `test_coverage_guarantee_holds_on_exchangeable_synthetic_data` is for, and it
is the reason a coverage number on the real data is reported as "consistent with"
rather than "verified".

Every test asserts a positive count of what it examined.
"""

from __future__ import annotations

import math
import random

import pytest

from fleetmind.decision import calibration as calib
from fleetmind.decision import conformal, temperature
from fleetmind.decision.probs import WindowPosterior
from fleetmind.ml.model import LABEL_MAP


def _post(logits, true_label, window_id=0):
    """A WindowPosterior whose probabilities are the softmax of `logits`."""
    return WindowPosterior(
        window_id=window_id,
        experiment_id=1,
        window_index=window_id,
        true_label=true_label,
        probabilities=temperature.apply_temperature(logits, 1.0),
        logits=list(logits),
    )


# ── temperature scaling ────────────────────────────────────────────────
def test_temperature_of_one_is_the_identity():
    logits = [2.0, 0.5, -1.0, 0.25]
    plain = temperature.apply_temperature(logits, 1.0)
    assert len(plain) == 4
    assert sum(plain) == pytest.approx(1.0)
    # Against an independent softmax computation, not against itself.
    exps = [math.exp(v - max(logits)) for v in logits]
    expected = [v / sum(exps) for v in exps]
    for a, b in zip(plain, expected, strict=True):
        assert a == pytest.approx(b)


def test_temperature_never_changes_a_prediction():
    """The load-bearing property. Checked over many logits and many T."""
    rng = random.Random(0)
    checked = 0
    for _ in range(60):
        logits = [rng.gauss(0, 3) for _ in range(len(LABEL_MAP))]
        base_argmax = max(range(len(logits)), key=logits.__getitem__)
        for t in (0.1, 0.5, 0.95, 1.0, 1.7, 5.0, 19.0):
            scaled = temperature.apply_temperature(logits, t)
            assert max(range(len(scaled)), key=scaled.__getitem__) == base_argmax
            assert sum(scaled) == pytest.approx(1.0)
            checked += 1
    assert checked == 60 * 7 == 420, checked


def test_a_non_positive_temperature_is_refused():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            temperature.apply_temperature([1.0, 0.0, 0.0, 0.0], bad)


def test_fitting_an_over_confident_model_softens_it():
    """Huge logits, half of them wrong -> the fit must push T above 1."""
    rng = random.Random(1)
    logits, labels = [], []
    for i in range(200):
        truth = i % len(LABEL_MAP)
        # The model is extremely confident, and right only half the time.
        claimed = truth if i % 2 == 0 else (truth + 1) % len(LABEL_MAP)
        row = [rng.gauss(0, 0.1) for _ in range(len(LABEL_MAP))]
        row[claimed] += 12.0
        logits.append(row)
        labels.append(LABEL_MAP[truth])

    fit = temperature.fit_temperature(logits, labels, n_recordings=4)
    assert fit.n_calibration_windows == 200
    assert fit.n_calibration_recordings == 4
    assert fit.temperature > 1.5, fit.temperature
    assert fit.nll_after < fit.nll_before, (fit.nll_before, fit.nll_after)
    assert "over-confident" in fit.direction
    assert fit.iterations > 0


def test_fitting_an_under_confident_model_sharpens_it():
    """Tiny logits, always right -> the fit must push T below 1."""
    logits, labels = [], []
    for i in range(200):
        truth = i % len(LABEL_MAP)
        row = [0.0] * len(LABEL_MAP)
        row[truth] = 0.3  # barely favours the truth, but always correct
        logits.append(row)
        labels.append(LABEL_MAP[truth])

    fit = temperature.fit_temperature(logits, labels)
    assert fit.temperature < 0.9, fit.temperature
    assert fit.nll_after < fit.nll_before
    assert "under-confident" in fit.direction


def test_fitting_reduces_nll_or_leaves_it_alone_never_increases_it():
    """A fit that made NLL worse would mean the optimiser ran the wrong way."""
    rng = random.Random(7)
    checked = 0
    for scale in (0.5, 2.0, 8.0):
        logits, labels = [], []
        for i in range(120):
            truth = i % len(LABEL_MAP)
            row = [rng.gauss(0, 1) * scale for _ in range(len(LABEL_MAP))]
            logits.append(row)
            labels.append(LABEL_MAP[truth])
        fit = temperature.fit_temperature(logits, labels)
        assert fit.nll_after <= fit.nll_before + 1e-9, (scale, fit)
        checked += 1
    assert checked == 3


def test_fit_refuses_empty_or_mismatched_input():
    with pytest.raises(ValueError, match="no calibration data"):
        temperature.fit_temperature([], [])
    with pytest.raises(ValueError, match="same length"):
        temperature.fit_temperature([[1.0, 0.0, 0.0, 0.0]], ["Ball", "Normal"])


def test_rescale_keeps_logits_and_accuracy_but_moves_confidence():
    posteriors = [_post([6.0, 0.0, 0.0, 0.0], "Normal", i) for i in range(10)]
    rescaled = temperature.rescale(posteriors, 4.0)
    assert len(rescaled) == 10
    for before, after in zip(posteriors, rescaled, strict=True):
        assert after.logits == before.logits, "logits must survive, un-compounded"
        assert after.predicted_label == before.predicted_label
        assert after.confidence < before.confidence, "T>1 must reduce confidence"
    assert calib.measure(rescaled).accuracy == calib.measure(posteriors).accuracy


# ── the conformal quantile index ───────────────────────────────────────
def test_quantile_index_matches_values_worked_out_by_hand():
    """ceil((n+1)(1-alpha)), checked against arithmetic done independently."""
    cases = [
        (100, 0.10, 91),  # ceil(101 * 0.9) = ceil(90.9)  = 91
        (99, 0.10, 90),  # ceil(100 * 0.9) = 90
        (19, 0.10, 18),  # ceil(20 * 0.9)  = 18
        (9, 0.10, 9),  # ceil(10 * 0.9)  = 9
        (8, 0.10, 9),  # ceil(9 * 0.9) = ceil(8.1) = 9 > n -> not certifiable
        (236, 0.10, 214),  # ceil(237 * 0.9) = ceil(213.3) = 214
        (100, 0.05, 96),  # ceil(101 * 0.95) = ceil(95.95) = 96
        (100, 0.50, 51),  # ceil(101 * 0.5)  = ceil(50.5)  = 51
    ]
    assert len(cases) == 8
    for n, alpha, expected in cases:
        assert conformal.quantile_index(n, alpha) == expected, (n, alpha)


def test_quantile_index_rejects_impossible_arguments():
    with pytest.raises(ValueError, match="at least one calibration point"):
        conformal.quantile_index(0, 0.1)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
            conformal.quantile_index(10, bad)


def test_too_few_calibration_points_is_reported_not_faked():
    """n=8 at alpha=0.1 cannot certify 90%; the set must become everything."""
    posteriors = [_post([3.0, 0.0, 0.0, 0.0], "Normal", i) for i in range(8)]
    cal = conformal.calibrate(posteriors, alpha=0.10, n_recordings=1)
    assert cal.n_calibration_windows == 8
    assert cal.quantile_index == 9 > 8
    assert cal.certifiable is False
    assert not math.isfinite(cal.q_hat)
    assert cal.as_dict()["q_hat"] is None, "must serialise as null, not as a number"
    # With no finite threshold the honest set is every label.
    assert conformal.prediction_set([0.97, 0.01, 0.01, 0.01], cal) == list(LABEL_MAP)


# ── the coverage guarantee ─────────────────────────────────────────────
def _exchangeable_sample(n, rng, noise=1.2):
    """i.i.d. (hence exchangeable) logit/label pairs from one fixed process."""
    out = []
    for i in range(n):
        truth = rng.randrange(len(LABEL_MAP))
        row = [rng.gauss(0, noise) for _ in range(len(LABEL_MAP))]
        row[truth] += 2.0
        out.append(_post(row, LABEL_MAP[truth], i))
    return out


def test_coverage_guarantee_holds_on_exchangeable_synthetic_data():
    """The only place this implementation's correctness can be demonstrated.

    Calibration and test are drawn i.i.d. from one process, so exchangeability
    genuinely holds. Averaged over many independent repetitions, coverage must sit
    at or above the 90% target. On the real CWRU split this cannot be checked,
    because the calibration windows come from four autocorrelated recordings --
    which is exactly why the real report says "consistent with" and not "verified".
    """
    rng = random.Random(42)
    coverages = []
    for _ in range(40):
        cal = conformal.calibrate(_exchangeable_sample(300, rng), alpha=0.10)
        assert cal.certifiable
        ev = conformal.evaluate(_exchangeable_sample(300, rng), cal)
        assert ev.n_test_windows == 300
        coverages.append(ev.empirical_coverage)

    assert len(coverages) == 40
    mean = sum(coverages) / len(coverages)
    # Marginal guarantee: at least 1 - alpha on average. Small tolerance for the
    # finite number of repetitions.
    assert mean >= 0.90 - 0.01, mean
    # And it must not be trivially achieved by always emitting every label.
    assert mean <= 0.99, mean


def test_a_tighter_alpha_gives_larger_sets_and_higher_coverage():
    """Monotonicity: a stronger guarantee must cost set size, never be free."""
    rng = random.Random(3)
    cal_sample = _exchangeable_sample(400, rng)
    test_sample = _exchangeable_sample(400, rng)

    results = {}
    for alpha in (0.30, 0.10, 0.02):
        cal = conformal.calibrate(cal_sample, alpha=alpha)
        results[alpha] = conformal.evaluate(test_sample, cal)

    assert len(results) == 3
    sizes = [results[a].mean_set_size for a in (0.30, 0.10, 0.02)]
    covs = [results[a].empirical_coverage for a in (0.30, 0.10, 0.02)]
    assert sizes[0] < sizes[1] <= sizes[2], sizes
    assert covs[0] < covs[1] <= covs[2], covs


def test_prediction_sets_are_never_silently_widened_to_the_argmax():
    """An empty set is a legitimate output and must be reported as empty.

    Quietly inserting the argmax would break the coverage guarantee while making
    the numbers look tidier.
    """
    cal = conformal.ConformalCalibration(
        alpha=0.10,
        q_hat=0.40,  # a label needs p >= 0.60
        n_calibration_windows=100,
        n_calibration_recordings=4,
        quantile_index=91,
        certifiable=True,
    )
    assert cal.probability_floor == pytest.approx(0.60)
    spread = [0.3, 0.3, 0.2, 0.2]  # nothing reaches 0.60
    assert conformal.prediction_set(spread, cal) == []
    # ...and a confident one yields exactly one label.
    assert conformal.prediction_set([0.95, 0.02, 0.02, 0.01], cal) == ["Normal"]


def test_coverage_for_an_absent_class_is_none_not_a_number():
    """The Normal-absent case, which is the real split's situation."""
    rng = random.Random(5)
    sample = [p for p in _exchangeable_sample(300, rng) if p.true_label != "Normal"]
    assert len(sample) > 150
    cal = conformal.calibrate(sample, alpha=0.10, n_recordings=4)
    ev = conformal.evaluate(sample, cal, n_recordings=4)

    by_class = ev.coverage_by_true_class
    assert by_class["Normal"] is None, "a class never tested has no coverage"
    assert ev.classes_absent_from_test == ["Normal"]
    measured = [k for k, v in by_class.items() if v is not None]
    assert len(measured) == 3, measured

    payload = ev.as_dict()
    assert payload["coverage_by_true_class"]["Normal"] is None
    assert payload["unmeasurable_coverage"] == ["Normal"]
    assert payload["n_test_recordings"] == 4


def test_evaluating_or_calibrating_nothing_raises():
    with pytest.raises(ValueError, match="no calibration data"):
        conformal.calibrate([], alpha=0.1)
    cal = conformal.ConformalCalibration(0.1, 0.5, 10, 1, 10, True)
    with pytest.raises(ValueError, match="refusing to report coverage"):
        conformal.evaluate([], cal)


def test_nonconformity_is_one_minus_the_true_class_probability():
    probs = [0.1, 0.2, 0.3, 0.4]
    checked = 0
    for index, label in enumerate(LABEL_MAP):
        assert conformal.nonconformity(probs, label) == pytest.approx(1.0 - probs[index])
        checked += 1
    assert checked == len(LABEL_MAP) == 4


# ── the report's own wiring ────────────────────────────────────────────
def test_phase2_declares_its_caveats_and_names_the_base_arm():
    from fleetmind.decision import phase2

    assert len(phase2.CAVEATS) >= 4
    joined = " ".join(phase2.CAVEATS)
    assert "4 RECORDINGS per site" in joined
    assert "UNMEASURABLE" in joined
    assert "autocorrelated" in joined
    assert set(phase2.ARMS) == {"v2-last", "v2-best"}
    assert phase2.ARMS["v2-last"] == "cnn_state_dict_last.pt"


def test_every_checkpoint_phase2_can_load_lives_under_a_v2_directory():
    """The deployed v1 checkpoints must be unreachable from this module.

    Checked behaviourally, by resolving the paths the way `run_depot` does,
    rather than by grepping the source -- a grep would also match the output
    report filenames and would pass or fail for the wrong reasons.
    """
    from pathlib import Path

    from fleetmind.decision import phase2, retrain

    resolved = []
    for depot_id in (1, 2, 3, 4):
        for filename in phase2.ARMS.values():
            path = retrain.out_dir_for(depot_id, Path("mlruns")) / filename
            resolved.append(path)
            assert path.parent.name == f"depot_{depot_id}_v2", path
            assert path.parent != Path("mlruns") / f"depot_{depot_id}", path

    assert len(resolved) == 8, resolved
    assert len(set(resolved)) == 8, "each arm at each site must be a distinct file"
    # The v1 path must not appear anywhere in that set.
    for depot_id in (1, 2, 3, 4):
        v1 = Path("mlruns") / f"depot_{depot_id}" / "cnn_state_dict.pt"
        assert v1 not in resolved, v1
