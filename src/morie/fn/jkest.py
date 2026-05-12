# morie.fn -- function file (hadesllm/morie)
"""Jackknife bias and variance estimation (Quenouille 1956, Tukey 1958)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["jackknife_estimator"]


def jackknife_estimator(x, statistic=None):
    """Leave-one-out jackknife bias and variance.

    For statistic ``T_hat`` of the full sample and ``T_{-i}`` of the
    leave-one-out samples:

        bias_jack = (n-1) * (mean(T_{-i}) - T_hat)
        var_jack  = (n-1)/n * sum_i (T_{-i} - mean(T_{-i}))^2
        T_jack    = n * T_hat - (n-1) * mean(T_{-i})

    Parameters
    ----------
    x : array-like
    statistic : callable, optional
        Default = sample mean.

    Returns
    -------
    RichResult
        Keys: estimate (T_jack), bias, se, var, n, method.

    References
    ----------
    Quenouille (1956), Tukey (1958), Efron & Tibshirani (1993) Ch. 11.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return RichResult(payload={"estimate": float("nan"), "bias": float("nan"),
                                   "se": float("nan"), "n": int(n),
                                   "method": "Jackknife (n<2)"})
    if statistic is None:
        statistic = np.mean
    T_hat = float(statistic(x))
    T_loo = np.array([statistic(np.delete(x, i)) for i in range(n)], dtype=float)
    T_bar = float(T_loo.mean())
    bias = (n - 1) * (T_bar - T_hat)
    T_jack = n * T_hat - (n - 1) * T_bar
    var_jack = (n - 1) / n * float(np.sum((T_loo - T_bar) ** 2))
    se = float(np.sqrt(var_jack))
    return RichResult(payload={
        "estimate": T_jack, "theta_hat": T_hat, "bias": float(bias),
        "var": var_jack, "se": se, "n": int(n),
        "method": "Jackknife (Quenouille 1956)",
    })


# CANONICAL TEST
# >>> x = [3.0, 5.0, 7.0, 9.0, 11.0]
# >>> res = jackknife_estimator(x)
# >>> assert abs(res["theta_hat"] - 7.0) < 1e-12
# >>> # bias of mean estimator is exactly 0 under jackknife
# >>> assert abs(res["bias"]) < 1e-12


def cheatsheet():
    return "jkest(x, statistic=mean): leave-one-out jackknife bias/var/SE."
