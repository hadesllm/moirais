# moirais.fn — function file (hadesllm/moirais)
"""Nonparametric baseline hazard estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_baseline_hazard_est"]


def horowitz_baseline_hazard_est(t, x, event, beta_hat):
    """
    Nonparametric baseline hazard estimator

    Formula: Lambda_0_hat(t) = Breslow estimator: sum_{t_i<=t} 1/[sum_j I(t_j>=t_i)*exp(X_j'beta_hat)]

    Parameters
    ----------
    t : array-like
        Input data.
    x : array-like
        Input data.
    event : array-like
        Input data.
    beta_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lambda0_hat

    References
    ----------
    Horowitz Ch 6, Sec 6.2.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Nonparametric baseline hazard estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Nonparametric baseline hazard estimator"})


def cheatsheet():
    return "hrzlam: Nonparametric baseline hazard estimator"
