# morie.fn -- function file (hadesllm/morie)
"""All models are wrong, but some are useful. -- George E. P. Box"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bbs_changepoint(y: np.ndarray, min_size: int = 2, penalty: float | None = None) -> DescriptiveResult:
    """
    Binary segmentation algorithm for multiple change-point detection.

    Recursively splits the series at the point that maximises the
    reduction in total residual sum of squares, stopping when no split
    exceeds the penalty.

    :param y: 1-D time series.
    :type y: numpy.ndarray
    :param min_size: Minimum segment length. Default 2.
    :type min_size: int
    :param penalty: BIC penalty per change point. Default ``2 * log(n)``.
    :type penalty: float or None
    :return: DescriptiveResult with detected change-point indices.
    :rtype: DescriptiveResult

    References
    ----------
    Scott A.J. & Knott M. (1974). A Cluster Analysis Method for
    Grouping Means in the Analysis of Variance. *Biometrics*, 30(3),
    507-512.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if penalty is None:
        penalty = 2.0 * np.log(n)

    def _cost(seg: np.ndarray) -> float:
        if len(seg) == 0:
            return 0.0
        return float(np.sum((seg - np.mean(seg)) ** 2))

    cps: list[int] = []

    def _split(start: int, end: int) -> None:
        if end - start < 2 * min_size:
            return
        seg = y[start:end]
        base_cost = _cost(seg)
        best_gain = 0.0
        best_idx = -1
        for t in range(start + min_size, end - min_size + 1):
            gain = base_cost - _cost(y[start:t]) - _cost(y[t:end])
            if gain > best_gain:
                best_gain = gain
                best_idx = t
        if best_gain > penalty and best_idx > 0:
            cps.append(best_idx)
            _split(start, best_idx)
            _split(best_idx, end)

    _split(0, n)
    cps_sorted = sorted(cps)
    return DescriptiveResult(
        name="bbs_changepoint",
        value=len(cps_sorted),
        extra={"changepoints": np.array(cps_sorted), "n": n, "penalty": penalty},
    )


cpbbs = bbs_changepoint


def cheatsheet() -> str:
    return "bbs_changepoint({}) -> Binary segmentation for change-point detection. 'I have the "
