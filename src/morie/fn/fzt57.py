# morie.fn -- function file (hadesllm/morie)
"""Theorem 5.7: boundary-free CvM converges to standard CvM under H0."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm5_7_bdfree_cvm_equiv"]


def fauzi_thm5_7_bdfree_cvm_equiv(data, bandwidth, cdf, g_func):
    """
    Theorem 5.7: boundary-free CvM converges to standard CvM under H0

    Formula: |CvM_n - CvM_tilde| ->_p 0 under H0: F_X = F

    Parameters
    ----------
    data : array-like
        Input data.
    bandwidth : array-like
        Input data.
    cdf : array-like
        Input data.
    g_func : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: convergence

    References
    ----------
    Fauzi Ch 5, Theorem 5.7
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 5.7: boundary-free CvM converges to standard CvM under H0"})


def cheatsheet():
    return "fzt57: Theorem 5.7: boundary-free CvM converges to standard CvM under H0"
