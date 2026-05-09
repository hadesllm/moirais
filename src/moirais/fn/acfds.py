# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""ACF-based distance between two signals."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def acf_dist(x1: np.ndarray, x2: np.ndarray, max_lag: int = 20) -> DescriptiveResult:
    """Compute distance between two signals based on their autocorrelation functions.

    :param x1: First 1-D input signal.
    :param x2: Second 1-D input signal.
    :param max_lag: Maximum lag for ACF comparison (default 20).
    :return: DescriptiveResult with ACF distance value.
    """
    from moirais._adaptive import acf_distance

    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    distance = acf_distance(x1, x2, max_lag=max_lag)
    return DescriptiveResult(name="acf_distance", value=float(distance))


acfds = acf_dist


def cheatsheet() -> str:
    return "acf_dist({}) -> ACF-based distance between two signals."
