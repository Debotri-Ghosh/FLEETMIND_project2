from fleetmind.ingestion.ingest import find_de_time_key
from fleetmind.ingestion.windowing import split_windows


def test_split_windows_exact_multiple():
    signal = list(range(16))
    windows = split_windows(signal, window_size=8)
    assert len(windows) == 2
    start, end, values = windows[0]
    assert (start, end) == (0, 8)
    assert values == list(range(8))


def test_split_windows_drops_partial():
    signal = list(range(10))
    windows = split_windows(signal, window_size=8)
    assert len(windows) == 1


def test_split_windows_empty():
    assert split_windows([], window_size=8) == []


def test_find_de_time_key():
    assert find_de_time_key({"X097_DE_time": [1], "other": [2]}) == "X097_DE_time"
    assert find_de_time_key({"nope": [1]}) is None
