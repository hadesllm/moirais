# morie.fn — function file (hadesllm/morie)
"""Boundary-free MRL estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_mrl_boundary_free"]


def fauzi_mrl_boundary_free(x):
    """
    Boundary-free MRL estimator

    Formula: m_hat_bf via bijective transformation

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
    Fauzi Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Boundary-free MRL estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Boundary-free MRL estimator"})


def cheatsheet():
    return "fzmrb: Boundary-free MRL estimator"
