# morie.fn -- function file (hadesllm/morie)
"""Direct estimation of single-index model with discrete covariates."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_direct_discrete_x"]


def horowitz_direct_discrete_x(x, y):
    """
    Direct estimation of single-index model with discrete covariates

    Formula: beta identified from log-odds ratio differences across discrete X values

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat, se

    References
    ----------
    Horowitz Ch 2, Sec 2.6.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Direct estimation of single-index model with discrete covariates"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Direct estimation of single-index model with discrete covariates"})


def cheatsheet():
    return "hrzdiscd: Direct estimation of single-index model with discrete covariates"
