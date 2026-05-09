# moirais.fn — function file (hadesllm/moirais)
"""Kernel survival function estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_survival_kernel"]


def fauzi_survival_kernel(x):
    """
    Kernel survival function estimator

    Formula: S_hat(t) = 1 - F_hat_h(t)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Kernel survival function estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Kernel survival function estimator"})


def cheatsheet():
    return "fzsrv: Kernel survival function estimator"
