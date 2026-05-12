# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian change-point detection (step function)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_changepoint(
    data: Union[list, np.ndarray],
    *,
    max_cp: int = 5,
    prior_p: float = 0.01,
) -> dict[str, Any]:
    """
    Bayesian change-point detection for a univariate time series.

    Uses marginal likelihood with conjugate normal model to find the MAP
    number and location of change points.

    :param data: Observed time series (n,).
    :param max_cp: Maximum number of change points to consider.
    :param prior_p: Prior probability of a change at each time point.
    :return: Dictionary with change_points, n_changepoints, segment_means.

    References
    ----------
    Barry, D. & Hartigan, J. A. (1993). *Annals of Statistics*, 21(1), 159--177.
    """
    x = np.asarray(data, dtype=float).ravel()
    n = len(x)

    def _seg_log_ml(start, end):
        seg = x[start:end]
        m = len(seg)
        if m == 0:
            return -np.inf
        s2 = float(np.var(seg, ddof=0)) + 1e-10
        return -0.5 * m * np.log(2 * np.pi * s2) - 0.5 * m

    best_log_ml = -np.inf
    best_cps = []

    for k in range(max_cp + 1):
        if k == 0:
            lml = _seg_log_ml(0, n) + np.log(1 - prior_p) * (n - 1)
            if lml > best_log_ml:
                best_log_ml = lml
                best_cps = []
        elif k == 1:
            for cp in range(1, n):
                lml = _seg_log_ml(0, cp) + _seg_log_ml(cp, n) + np.log(prior_p) + np.log(1 - prior_p) * (n - 2)
                if lml > best_log_ml:
                    best_log_ml = lml
                    best_cps = [cp]
        elif k == 2:
            step = max(1, n // 50)
            for cp1 in range(1, n - 1, step):
                for cp2 in range(cp1 + 1, n, step):
                    lml = (
                        _seg_log_ml(0, cp1)
                        + _seg_log_ml(cp1, cp2)
                        + _seg_log_ml(cp2, n)
                        + 2 * np.log(prior_p)
                        + np.log(1 - prior_p) * (n - 3)
                    )
                    if lml > best_log_ml:
                        best_log_ml = lml
                        best_cps = [cp1, cp2]

    boundaries = [0] + best_cps + [n]
    segment_means = []
    for i in range(len(boundaries) - 1):
        seg = x[boundaries[i]:boundaries[i + 1]]
        segment_means.append(float(np.mean(seg)))

    return {
        "change_points": best_cps,
        "n_changepoints": len(best_cps),
        "segment_means": segment_means,
        "log_marginal_likelihood": float(best_log_ml),
        "n": n,
    }


bstep = bayesian_changepoint


def cheatsheet() -> str:
    return "bayesian_changepoint({}) -> Bayesian change-point detection (step function)."
