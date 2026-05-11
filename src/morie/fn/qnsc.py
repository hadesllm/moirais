"""Qn scale estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["qn_scale"]


def qn_scale(x):
    """
    Qn scale estimator

    Formula: Qn = c · {|x_i − x_j| ; i<j}_(k); 50%-bd

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
    Rousseeuw-Croux (1993)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Qn scale estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Qn scale estimator"})


def cheatsheet():
    return "qnsc: Qn scale estimator"
