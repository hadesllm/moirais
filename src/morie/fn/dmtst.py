# morie.fn — function file (hadesllm/morie)
"""Diebold-Mariano forecast comparison test."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def dm_test(actual: np.ndarray, forecast1: np.ndarray, forecast2: np.ndarray, h: int = 1, loss: str = "squared", cdf=None) -> DescriptiveResult:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Tests H0: both forecasts have equal accuracy vs H1: they differ.

    :param actual: 1-D actual values.
    :param forecast1: 1-D forecasts from model 1.
    :param forecast2: 1-D forecasts from model 2.
    :param h: Forecast horizon (for HAC variance). Default 1.
    :param loss: 'squared' or 'absolute'. Default 'squared'.
    :return: DescriptiveResult with DM statistic and p-value.
    :raises ValueError: On length mismatch.

    References
    ----------
    Diebold F.X. & Mariano R.S. (1995). Comparing predictive accuracy.
    *JBES*, 13(3), 253-263.
    """
    a = np.asarray(actual, dtype=float).ravel()
    f1 = np.asarray(forecast1, dtype=float).ravel()
    f2 = np.asarray(forecast2, dtype=float).ravel()
    if not (len(a) == len(f1) == len(f2)):
        raise ValueError("All arrays must have same length.")
    n = len(a)
    if n < 5:
        raise ValueError(f"Need at least 5 observations, got {n}.")
    if loss == "squared":
        d = (a - f1) ** 2 - (a - f2) ** 2
    elif loss == "absolute":
        d = np.abs(a - f1) - np.abs(a - f2)
    else:
        raise ValueError(f"loss must be 'squared' or 'absolute', got '{loss}'.")
    d_bar = float(np.mean(d))
    gamma0 = float(np.sum((d - d_bar) ** 2) / n)
    var_d = gamma0
    for k in range(1, h):
        if k < n:
            gamma_k = float(np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / n)
            var_d += 2 * gamma_k
    var_d = max(var_d / n, 1e-15)
    dm_stat = d_bar / np.sqrt(var_d)
    p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return DescriptiveResult(
        name="dm_test",
        value=float(dm_stat),
        extra={
            "dm_statistic": float(dm_stat),
            "p_value": float(p_val),
            "mean_loss_diff": d_bar,
            "loss": loss,
            "h": h,
            "n": n,
        },
    )


dmtst = dm_test


def cheatsheet() -> str:
    return "dm_test({}) -> Diebold-Mariano forecast comparison test."
