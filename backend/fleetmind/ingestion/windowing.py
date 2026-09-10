"""Signal windowing — the pure, unit-testable core of the ingestion pipeline."""

from collections.abc import Sequence


def split_windows(
    signal: Sequence[float], window_size: int = 2048
) -> list[tuple[int, int, list[float]]]:
    """Chop a continuous signal into fixed-length non-overlapping windows.

    Returns a list of (start_sample, end_sample, values) tuples. The final
    partial window is dropped (standard practice in CWRU literature).
    """
    n_windows = len(signal) // window_size
    return [
        (
            w * window_size,
            (w + 1) * window_size,
            list(signal[w * window_size : (w + 1) * window_size]),
        )
        for w in range(n_windows)
    ]
