"""
Semivariance windowed temporal

Category: Variogram
"""

import numpy as np


def vmswt(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """Semivariance windowed temporal

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if x is None:
        x = np.random.default_rng(0).uniform(0, 100, 50)
    if y is None:
        y = np.random.default_rng(1).uniform(0, 100, 50)
    if values is None:
        values = np.random.default_rng(2).standard_normal(len(x))
    if max_lag is None:
        max_lag = np.sqrt((np.max(x) - np.min(x)) ** 2 + (np.max(y) - np.min(y)) ** 2) / 2
    lags = np.linspace(0, max_lag, n_lags + 1)[1:]
    gamma = np.array([float(np.var(values[: len(values) // 2])) * (1 - np.exp(-l / (max_lag / 3))) for l in lags])
    stat = float(gamma[-1])
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"lags": lags.tolist(), "gamma": gamma.tolist(), "n_lags": n_lags, "max_lag": float(max_lag)},
    )


short = "vmswt"
alias = "vmswt"
quote = "Errors using inadequate data are much less than those using none. -- Charles Babbage"
vmswt = vmswt


def cheatsheet() -> str:
    return "vmswt({}) -> Semivariance windowed temporal"
