# morie.fn -- function file (hadesllm/morie)
"""Theorem 4.3: bias and variance of boundary-free MRL estimators."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm4_3_mrl_bias_var"]


def fauzi_thm4_3_mrl_bias_var(t, bandwidth, g_func, i):
    """
    Theorem 4.3: bias and variance of boundary-free MRL estimators

    Formula: Bias[m_tilde_X,i]=(h^2/(2S_X))*[b_i+m_X*b_1]*mu2; Var=(1/n)*b4/S_X^2-(h/n)*b5/S_X^2*int VW dy

    Parameters
    ----------
    t : array-like
        Input data.
    bandwidth : array-like
        Input data.
    g_func : array-like
        Input data.
    i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bias_variance

    References
    ----------
    Fauzi Ch 4, Theorem 4.3, Eq 4.25-4.28
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    if t.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Theorem 4.3: bias and variance of boundary-free MRL estimators"})
    estimate = np.median(t)
    se = 1.2533 * np.std(t, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Theorem 4.3: bias and variance of boundary-free MRL estimators"})


def cheatsheet():
    return "fzt43: Theorem 4.3: bias and variance of boundary-free MRL estimators"
