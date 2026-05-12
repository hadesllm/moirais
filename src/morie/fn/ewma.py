# morie.fn -- function file (hadesllm/morie)
"""EWMA control chart for outbreak detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ewma_detect(
    counts: np.ndarray | list,
    *,
    lambda_: float = 0.3,
    L: float = 3.0,
    baseline_mean: float | None = None,
    baseline_sd: float | None = None,
) -> DescriptiveResult:
    """
    Exponentially Weighted Moving Average control chart for surveillance.

    Parameters
    ----------
    counts : array-like
        Time series of counts.
    lambda_ : float
        Smoothing parameter (0 < lambda_ <= 1).
    L : float
        Control limit multiplier.
    baseline_mean, baseline_sd : float, optional
        If None, estimated from first half of data.

    Returns
    -------
    DescriptiveResult
        extra has 'ewma', 'ucl', 'lcl', 'alarm_indices'.

    References
    ----------
    Lucas, J. M., & Saccucci, M. S. (1990). EWMA control schemes.
    *Technometrics*, 32(1), 1-12.
    """
    x = np.asarray(counts, dtype=float)
    if len(x) < 4:
        raise ValueError("Need at least 4 observations.")
    if not (0 < lambda_ <= 1):
        raise ValueError("lambda_ must be in (0, 1].")

    if baseline_mean is None:
        half = len(x) // 2
        baseline_mean = float(np.mean(x[:half]))
    if baseline_sd is None:
        half = len(x) // 2
        baseline_sd = float(np.std(x[:half], ddof=1))

    n = len(x)
    ewma = np.zeros(n)
    ucl = np.zeros(n)
    lcl = np.zeros(n)
    ewma[0] = baseline_mean

    for i in range(n):
        if i == 0:
            ewma[i] = lambda_ * x[i] + (1 - lambda_) * baseline_mean
        else:
            ewma[i] = lambda_ * x[i] + (1 - lambda_) * ewma[i - 1]
        factor = lambda_ / (2 - lambda_) * (1 - (1 - lambda_) ** (2 * (i + 1)))
        sigma_ewma = baseline_sd * np.sqrt(factor)
        ucl[i] = baseline_mean + L * sigma_ewma
        lcl[i] = baseline_mean - L * sigma_ewma

    alarms = np.where(ewma > ucl)[0]

    return DescriptiveResult(
        name="EWMA",
        value=float(len(alarms)),
        extra={
            "ewma": ewma,
            "ucl": ucl,
            "lcl": lcl,
            "alarm_indices": alarms.tolist(),
        },
    )


ewma = ewma_detect


def cheatsheet() -> str:
    return "ewma_detect({}) -> EWMA control chart for outbreak detection."
