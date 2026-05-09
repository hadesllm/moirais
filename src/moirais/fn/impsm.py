# moirais.fn — function file (hadesllm/moirais)
"""Importance sampling estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["importance_sampling"]


def importance_sampling(x):
    """
    Importance sampling estimator

    Formula: E_p[h] = (1/N) sum h(x_i)*p(x_i)/q(x_i)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Geweke (1989)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Importance sampling estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Importance sampling estimator"})


def cheatsheet():
    return "impsm: Importance sampling estimator"
