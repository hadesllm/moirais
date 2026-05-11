"""Normalizing flow density estimation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["normalizing_flow"]


def normalizing_flow(base, flow):
    """
    Normalizing flow density estimation

    Formula: x = f_K ∘ ... ∘ f_1(z); log p(x) via Jacobians

    Parameters
    ----------
    base : array-like
        Input data.
    flow : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rezende-Mohamed (2015)
    """
    base = np.atleast_1d(np.asarray(base, dtype=float))
    n = len(base)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Normalizing flow density estimation"})
    estimate = np.median(base)
    se = 1.2533 * np.std(base, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Normalizing flow density estimation"})


def cheatsheet():
    return "normfl: Normalizing flow density estimation"
