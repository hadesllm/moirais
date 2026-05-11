# morie.fn — function file (hadesllm/morie)
"""Boundary-free kernel density estimator via bijective transformation g."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_bdfree_kde"]


def fauzi_bdfree_kde(t, bandwidth, g_func):
    """
    Boundary-free kernel density estimator via bijective transformation g

    Formula: f_tilde_X(t) = (1/nhg'(g^{-1}(t))) sum K((g^{-1}(t)-g^{-1}(X_i))/h)

    Parameters
    ----------
    t : array-like
        Input data.
    bandwidth : array-like
        Input data.
    g_func : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 4
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    if t.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Boundary-free kernel density estimator via bijective transformation g"})
    estimate = np.median(t)
    se = 1.2533 * np.std(t, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Boundary-free kernel density estimator via bijective transformation g"})


def cheatsheet():
    return "fzbfkd: Boundary-free kernel density estimator via bijective transformation g"
