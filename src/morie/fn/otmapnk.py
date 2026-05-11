"""Stub for neural-Kantorovich monotone map estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_map_neural_kantorovich"]


def ot_map_neural_kantorovich(X, Y, epochs):
    """
    Stub for neural-Kantorovich monotone map estimator

    Formula: Train ICNN φ; T = ∇φ

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    epochs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: phi, T_grid

    References
    ----------
    Makkuva-Taghvaei-Lee-Oh (2020)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Stub for neural-Kantorovich monotone map estimator"})
    estimate = np.median(X)
    se = 1.2533 * np.std(X, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Stub for neural-Kantorovich monotone map estimator"})


def cheatsheet():
    return "otmapnk: Stub for neural-Kantorovich monotone map estimator"
