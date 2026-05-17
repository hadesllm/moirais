"""
K-Bessel variogram model

Category: Variogram
"""

import numpy as np


def vmkbs(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """K-Bessel variogram model

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


short = "vmkbs"
alias = "vmkbs"
quote = "Logic is the foundation of all certain knowledge. -- Leonhard Euler"
vmkbs = vmkbs


def cheatsheet() -> str:
    return "vmkbs({}) -> K-Bessel variogram model"
