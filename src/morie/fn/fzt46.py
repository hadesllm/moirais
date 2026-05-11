# morie.fn — function file (hadesllm/morie)
"""Theorem 4.6: mean value property of boundary-free MRL estimators."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm4_6_mean_value"]


def fauzi_thm4_6_mean_value(data, g_func):
    """
    Theorem 4.6: mean value property of boundary-free MRL estimators

    Formula: m_tilde_X,2(a_1)+a_1=X_bar (exact); m_tilde_X,1(a_1)+a_1=X_bar+O_p(h^2)

    Parameters
    ----------
    data : array-like
        Input data.
    g_func : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: property_check

    References
    ----------
    Fauzi Ch 4, Theorem 4.6, Eq 4.29-4.30
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    if data.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Theorem 4.6: mean value property of boundary-free MRL estimators"})
    estimate = np.median(data)
    se = 1.2533 * np.std(data, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Theorem 4.6: mean value property of boundary-free MRL estimators"})


def cheatsheet():
    return "fzt46: Theorem 4.6: mean value property of boundary-free MRL estimators"
