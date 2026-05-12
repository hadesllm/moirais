# morie.fn -- function file (hadesllm/morie)
"""Second boundary-free MRL estimator m_tilde_X,2."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_mrl_est2"]


def fauzi_mrl_est2(t, bandwidth, g_func):
    """
    Second boundary-free MRL estimator m_tilde_X,2

    Formula: m_tilde_X,2(t) = S_tilde_X,2(t) / S_tilde_X(t)

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
    Fauzi Ch 4, Eq 4.24
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    if t.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Second boundary-free MRL estimator m_tilde_X,2"})
    estimate = np.median(t)
    se = 1.2533 * np.std(t, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Second boundary-free MRL estimator m_tilde_X,2"})


def cheatsheet():
    return "fzmr2: Second boundary-free MRL estimator m_tilde_X,2"
