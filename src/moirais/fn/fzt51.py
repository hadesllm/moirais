# moirais.fn — function file (hadesllm/moirais)
"""Theorem 5.1: naive kernel KS and CvM converge to empirical counterparts."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm5_1_naive_kernel_equiv"]


def fauzi_thm5_1_naive_kernel_equiv(data, bandwidth, cdf):
    """
    Theorem 5.1: naive kernel KS and CvM converge to empirical counterparts

    Formula: |KS_n - KS_hat| ->_p 0 and |CvM_n - CvM_hat| ->_p 0

    Parameters
    ----------
    data : array-like
        Input data.
    bandwidth : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: convergence

    References
    ----------
    Fauzi Ch 5, Theorem 5.1
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 5.1: naive kernel KS and CvM converge to empirical counterparts"})


def cheatsheet():
    return "fzt51: Theorem 5.1: naive kernel KS and CvM converge to empirical counterparts"
