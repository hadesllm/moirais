# morie.fn — function file (hadesllm/morie)
"""Luck is what happens when preparation meets opportunity. — Seneca"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def pelt_changepoint(y: np.ndarray, penalty: str = "bic", min_size: int = 2) -> DescriptiveResult:
    r"""
    Pruned Exact Linear Time (PELT) algorithm for change-point detection.

    Finds the exact segmentation that minimises cost + penalty using
    dynamic programming with pruning:

    .. math::

        F(t) = \\min_{s \\in R_t} \\bigl[ F(s) + C(y_{s+1:t}) + \\beta \\bigr]

    :param y: 1-D time series.
    :type y: numpy.ndarray
    :param penalty: ``"bic"`` for :math:`\\log(n)` or a float. Default ``"bic"``.
    :type penalty: str or float
    :param min_size: Minimum segment length. Default 2.
    :type min_size: int
    :return: DescriptiveResult with optimal change points.
    :rtype: DescriptiveResult

    References
    ----------
    Killick R., Fearnhead P. & Eckley I.A. (2012). Optimal detection of
    changepoints with a linear computational cost. *Journal of the
    American Statistical Association*, 107(500), 1590-1598.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if isinstance(penalty, str) and penalty.lower() == "bic":
        pen = np.log(n)
    else:
        pen = float(penalty)

    cumsum = np.zeros(n + 1)
    cumsum2 = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(y)
    cumsum2[1:] = np.cumsum(y**2)

    def _cost(s: int, t: int) -> float:
        length = t - s
        if length <= 0:
            return 0.0
        sm = cumsum[t] - cumsum[s]
        sm2 = cumsum2[t] - cumsum2[s]
        return float(sm2 - sm**2 / length)

    F = np.full(n + 1, np.inf)
    F[0] = -pen
    cp_prev = np.zeros(n + 1, dtype=int)
    R = {0}

    for t in range(min_size, n + 1):
        candidates = {}
        for s in R:
            if t - s >= min_size:
                candidates[s] = F[s] + _cost(s, t) + pen
        if not candidates:
            F[t] = F[t - 1]
            cp_prev[t] = cp_prev[t - 1]
            R.add(t - min_size) if t - min_size >= 0 else None
            continue
        best_s = min(candidates, key=candidates.get)
        F[t] = candidates[best_s]
        cp_prev[t] = best_s
        R = {s for s in R if t - s >= min_size and F[s] + _cost(s, t) <= F[t]}
        R.add(t - min_size) if t - min_size >= 0 else None

    cps: list[int] = []
    idx = n
    while idx > 0:
        prev = cp_prev[idx]
        if prev > 0:
            cps.append(prev)
        idx = prev
    cps_sorted = sorted(cps)
    return DescriptiveResult(
        name="pelt_changepoint",
        value=len(cps_sorted),
        extra={"changepoints": np.array(cps_sorted), "n": n, "penalty": pen},
    )


cpelt = pelt_changepoint


def cheatsheet() -> str:
    return "pelt_changepoint({}) -> PELT change-point detection. 'Power! Unlimited power!' -- Da"
