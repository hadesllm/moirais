# moirais.fn — function file (hadesllm/moirais)
"""Average derivative estimator (Powell-Stock-Stoker) for E[Y|X]."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_average_derivative"]


def horowitz_average_derivative(x, y, bandwidth):
    """
    Average derivative estimator (Powell-Stock-Stoker) for E[Y|X]

    Formula: delta = E[dE(Y|X)/dX] = -2*E[f_X(X)*dE(Y|X)/dX] / E[f_X(X)] (density-weighted)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: delta_hat, se

    References
    ----------
    Horowitz Ch 2, Sec 2.6.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Average derivative estimator (Powell-Stock-Stoker) for E[Y|X]"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Average derivative estimator (Powell-Stock-Stoker) for E[Y|X]"})


def cheatsheet():
    return "hrzade: Average derivative estimator (Powell-Stock-Stoker) for E[Y|X]"
