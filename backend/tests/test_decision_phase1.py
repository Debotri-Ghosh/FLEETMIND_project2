"""Phase 1: the three-way split, best-epoch selection, and the v1/v2 comparison.

The property that matters most here is **the test set is never trained on, by any
arm**. If it were, the whole v1-vs-v2 comparison would be meaningless, and the
leak would be completely invisible in the output — the numbers would simply look
better. So it is asserted structurally rather than trusted.

The second property is that **best-epoch selection reads the selection set, not
the test set**. Selecting an epoch on the test set would turn the reported test
accuracy into an optimistic estimate of itself, which is the subtler form of the
bug Phase 0 found.

Every test asserts a positive count of what it examined.
"""

from __future__ import annotations

import pytest

# The synthetic-archive fixture and its constants now live in conftest.py, so the
# Phase 1 and Phase 3 tests share one definition of CWRU's recording structure.
from conftest import WINDOWS_PER_RECORDING  # noqa: E402

from fleetmind.decision import compare, retrain, split3


# ── the split's structure ──────────────────────────────────────────────
def _desc(train_ids, selection_ids, test_ids, **counts):
    return split3.Split3Description(
        depot_id=1,
        train_experiment_ids=list(train_ids),
        selection_experiment_ids=list(selection_ids),
        test_experiment_ids=list(test_ids),
        train_class_counts=counts.get("train", {"OuterRace": 10, "Normal": 5}),
        selection_class_counts=counts.get("selection", {"OuterRace": 4}),
        test_class_counts=counts.get("test", {"OuterRace": 6, "Ball": 3}),
    )


def test_a_split_with_any_overlap_is_rejected_at_construction():
    """Three separate overlap shapes, each named in the error."""
    cases = [
        ([1, 2, 3], [3, 4], [5, 6], "train&selection"),
        ([1, 2, 3], [4, 5], [3, 6], "train&test"),
        ([1, 2], [3, 4], [4, 5], "selection&test"),
    ]
    assert len(cases) == 3
    for train, selection, test, expected in cases:
        with pytest.raises(ValueError, match="must be disjoint"):
            _desc(train, selection, test)
        # and the message names which pair overlapped
        try:
            _desc(train, selection, test)
        except ValueError as exc:
            assert expected in str(exc), (expected, str(exc))


def test_an_empty_set_is_rejected():
    checked = 0
    for train, selection, test in (([], [1], [2]), ([1], [], [2]), ([1], [2], [])):
        with pytest.raises(ValueError, match="must be non-empty"):
            _desc(train, selection, test)
        checked += 1
    assert checked == 3


def test_a_valid_split_reports_floors_and_absent_classes():
    d = _desc(
        [1, 2],
        [3],
        [4],
        train={"OuterRace": 10, "Ball": 6, "Normal": 4},
        selection={"OuterRace": 6, "Ball": 2},
        test={"OuterRace": 8, "Ball": 2},
    )
    assert d.classes_absent_from_selection == ["Normal"]
    assert d.classes_absent_from_test == ["Normal"]
    assert d.test_majority_share == pytest.approx(0.8)
    assert d.selection_majority_share == pytest.approx(0.75)

    payload = d.as_dict()
    assert payload["recordings"] == {"train": 2, "selection": 1, "test": 1}
    assert payload["windows"] == {"train": 20, "selection": 8, "test": 10}
    assert "trained on by no arm" not in payload["derivation"]  # wording lives in compare.py
    assert "test = v1's original held-out recordings" in payload["derivation"]


# ── the real split, against the live archive ───────────────────────────
@pytest.mark.parametrize("depot_id", [1, 2, 3, 4])
def test_the_real_three_way_split_is_disjoint_and_test_matches_v1s_holdout(
    depot_id, archive_session
):
    """The leak guard, on real data, per site.

    `test` must be byte-identical to the recordings v1 held out -- that is the
    only reason v1 can be scored on it fairly.
    """
    from fleetmind.decision import holdout

    outer = holdout.split_for_depot(archive_session, depot_id)
    split = split3.three_way_split(archive_session, depot_id)

    assert split.test_experiment_ids == sorted(outer.holdout_experiment_ids)
    assert len(split.test_experiment_ids) > 0

    # train + selection must exactly partition v1's training recordings: nothing
    # invented, nothing dropped.
    assert sorted(split.train_experiment_ids + split.selection_experiment_ids) == sorted(
        outer.train_experiment_ids
    )

    ids = split.train_experiment_ids + split.selection_experiment_ids + split.test_experiment_ids
    assert len(ids) == len(set(ids)) == 16, "each site has 16 recordings, each used once"
    assert len(split.train_experiment_ids) >= 1
    assert len(split.selection_experiment_ids) >= 1


@pytest.mark.parametrize("depot_id", [1, 2, 3, 4])
def test_no_window_appears_in_more_than_one_of_the_three_sets(depot_id, archive_session):
    """Recording-level disjointness is only useful if it implies window-level."""
    split = split3.three_way_split(archive_session, depot_id)
    sets = {
        name: {w[0] for w in split3.windows_for(archive_session, ids)}
        for name, ids in (
            ("train", split.train_experiment_ids),
            ("selection", split.selection_experiment_ids),
            ("test", split.test_experiment_ids),
        )
    }
    assert all(len(v) > 0 for v in sets.values()), {k: len(v) for k, v in sets.items()}
    assert not sets["train"] & sets["selection"]
    assert not sets["train"] & sets["test"]
    assert not sets["selection"] & sets["test"]
    total = sum(len(v) for v in sets.values())
    assert total == len(sets["train"] | sets["selection"] | sets["test"])
    assert total == 16 * WINDOWS_PER_RECORDING, total


def test_the_single_normal_recording_stays_in_training_at_every_site(archive_session):
    """A limitation, pinned so it cannot be mistaken for an oversight later.

    Normal is absent from BOTH selection and test at every site, so best-epoch
    selection is a three-class criterion and a temperature fitted on the
    selection set would never see a healthy machine.
    """
    checked = 0
    for depot_id in (1, 2, 3, 4):
        split = split3.three_way_split(archive_session, depot_id)
        assert "Normal" in split.train_class_counts, depot_id
        assert "Normal" not in split.selection_class_counts, depot_id
        assert "Normal" not in split.test_class_counts, depot_id
        checked += 1
    assert checked == 4


# ── best-epoch selection ───────────────────────────────────────────────
def test_retraining_writes_to_v2_and_never_to_the_deployed_directory():
    """`mlruns/depot_N` must be untouchable: it produced all 4,367 decisions."""
    from pathlib import Path

    for depot_id in (1, 2, 3, 4):
        out = retrain.out_dir_for(depot_id, Path("mlruns"))
        assert out.name == f"depot_{depot_id}_v2"
        assert out != Path("mlruns") / f"depot_{depot_id}"


def test_retrain_selects_the_argmax_epoch_and_prefers_the_earlier_of_a_tie(
    archive_session, tmp_path
):
    """A 2-epoch real run, checked against its own recorded history.

    Short on purpose -- this asserts the selection *logic*, not model quality,
    and a long run would make the suite slow for no extra coverage.
    """
    report = retrain.retrain_depot(
        archive_session, 2, epochs=2, seed=7, mlruns=tmp_path, write=True
    )

    history = report["selection_accuracy_history"]
    assert len(history) == 2, history
    best = max(history)
    assert report["best_selection_accuracy"] == pytest.approx(best)
    # Earliest index achieving the max -- strict `>` must not take a later tie.
    assert report["best_epoch"] == history.index(best) + 1
    assert report["last_selection_accuracy"] == pytest.approx(history[-1])

    out = tmp_path / "depot_2_v2"
    assert (out / "cnn_state_dict.pt").exists()
    assert (out / "cnn_state_dict_last.pt").exists()
    assert (out / "metrics.json").exists()

    # The two arms must be loadable and the split recorded alongside them.
    assert report["split"]["recordings"]["test"] == 4
    assert report["split"]["classes_absent_from_test"] == ["Normal"]


def test_retrain_refuses_a_split_whose_test_recordings_leaked(
    archive_session, monkeypatch, tmp_path
):
    """The assertion inside retrain_depot is a real guard, not decoration.

    Forces a split where test overlaps train and checks training refuses to
    start. Without this, the guard could be dead code and nothing would say so.
    """
    real = split3.three_way_split(archive_session, 1)
    leaked = split3.Split3Description.__new__(split3.Split3Description)
    object.__setattr__(leaked, "depot_id", 1)
    object.__setattr__(leaked, "train_experiment_ids", real.train_experiment_ids)
    object.__setattr__(leaked, "selection_experiment_ids", real.selection_experiment_ids)
    # The leak: a training recording also listed as test.
    object.__setattr__(leaked, "test_experiment_ids", real.train_experiment_ids[:1])
    object.__setattr__(leaked, "train_class_counts", real.train_class_counts)
    object.__setattr__(leaked, "selection_class_counts", real.selection_class_counts)
    object.__setattr__(leaked, "test_class_counts", real.test_class_counts)

    monkeypatch.setattr(split3, "three_way_split", lambda *a, **k: leaked)
    with pytest.raises(AssertionError, match="leaked into training or selection"):
        retrain.retrain_depot(archive_session, 1, epochs=1, mlruns=tmp_path, write=False)


# ── the comparison ─────────────────────────────────────────────────────
def test_compare_names_three_distinct_arms_pointing_at_three_distinct_files():
    from pathlib import Path

    arms = compare.arms_for(3, Path("mlruns"))
    assert len(arms) == 3
    assert len(set(arms.values())) == 3, arms

    v1 = next(p for name, p in arms.items() if name.startswith("v1"))
    assert v1.parent.name == "depot_3", v1
    assert all(p.parent.name == "depot_3_v2" for name, p in arms.items() if name.startswith("v2"))
    # The two v2 arms must be different files from the same run.
    v2 = [p for name, p in arms.items() if name.startswith("v2")]
    assert {p.name for p in v2} == {"cnn_state_dict.pt", "cnn_state_dict_last.pt"}


def test_compare_fails_loudly_when_v2_has_not_been_trained(archive_session, tmp_path):
    """Missing checkpoints must raise, not be silently skipped.

    A comparison that quietly dropped an arm would report a one-arm 'comparison'
    that still looked like a result.
    """
    (tmp_path / "depot_1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="run `python -m fleetmind.decision.retrain`"):
        compare.compare_depot(archive_session, 1, tmp_path)
