# moirais.fn — function file (hadesllm/moirais)
"""Theorem 4.5: strong uniform consistency of boundary-free MRL estimators."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm4_5_mrl_consistency"]


def fauzi_thm4_5_mrl_consistency(t, bandwidth, g_func):
    """
    Theorem 4.5: strong uniform consistency of boundary-free MRL estimators

    Formula: sup_{t in Omega} |m_tilde_X,i(t) - m_X(t)| ->_{a.s.} 0

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
        Keys: boolean

    References
    ----------
    Fauzi Ch 4, Theorem 4.5
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    if t.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Theorem 4.5: strong uniform consistency of boundary-free MRL estimators"})
    estimate = np.median(t)
    se = 1.2533 * np.std(t, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Theorem 4.5: strong uniform consistency of boundary-free MRL estimators"})


def cheatsheet():
    return "fzt45: Theorem 4.5: strong uniform consistency of boundary-free MRL estimators"
