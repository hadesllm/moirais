# morie.fn — function file (hadesllm/morie)
"""Poisson change-point model. 'I sense great fear in you.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def changepoint_poisson(counts: np.ndarray) -> DescriptiveResult:
    """
    Detect a single change point in Poisson count data via maximum
    likelihood.

    Assumes :math:`Y_t \\sim \\mathrm{Pois}(\\lambda_1)` for
    :math:`t \\le \\tau` and :math:`Y_t \\sim \\mathrm{Pois}(\\lambda_2)`
    for :math:`t > \\tau`. Finds :math:`\\tau` maximising the profile
    log-likelihood.

    :param counts: 1-D array of non-negative integer counts.
    :type counts: numpy.ndarray
    :return: DescriptiveResult with change point and rate estimates.
    :rtype: DescriptiveResult
    :raises ValueError: If counts contain negative values.

    References
    ----------
    Worsley K.J. (1986). Confidence regions and tests for a change-point
    in a sequence of exponential family random variables. *Biometrika*,
    73(1), 91-104.
    """
    y = np.asarray(counts, dtype=float).ravel()
    n = len(y)
    if np.any(y < 0):
        raise ValueError("Counts must be non-negative.")
    if n < 4:
        raise ValueError(f"Need >= 4 observations, got {n}.")
    cumsum = np.cumsum(y)
    total = cumsum[-1]
    best_ll = -np.inf
    best_tau = 1
    for tau in range(2, n - 1):
        s1 = cumsum[tau - 1]
        s2 = total - s1
        lam1 = s1 / tau
        lam2 = s2 / (n - tau)
        if lam1 <= 0 or lam2 <= 0:
            continue
        ll = s1 * np.log(lam1) - tau * lam1 + s2 * np.log(lam2) - (n - tau) * lam2
        if ll > best_ll:
            best_ll = ll
            best_tau = tau
    lam1 = float(cumsum[best_tau - 1] / best_tau)
    lam2 = float((total - cumsum[best_tau - 1]) / (n - best_tau))
    null_lam = total / n
    null_ll = total * np.log(null_lam) - n * null_lam if null_lam > 0 else -np.inf
    lr_stat = 2.0 * (best_ll - null_ll)
    return DescriptiveResult(
        name="changepoint_poisson",
        value=best_tau,
        extra={
            "changepoint": best_tau,
            "lambda1": lam1,
            "lambda2": lam2,
            "lr_statistic": float(lr_stat),
            "log_likelihood": float(best_ll),
        },
    )


cppgm = changepoint_poisson


def cheatsheet() -> str:
    return "changepoint_poisson({}) -> Poisson change-point model. 'I sense great fear in you.' -- "
