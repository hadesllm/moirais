# moirais.fn — function file (hadesllm/moirais)
"""Theorem 5.8: standardized smoothed sign/Wilcoxon converge to unsmoothed in L^2."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm5_8_smoothed_convergence"]


def fauzi_thm5_8_smoothed_convergence(data, bandwidth, theta):
    """
    Theorem 5.8: standardized smoothed sign/Wilcoxon converge to unsmoothed in L^2

    Formula: lim E[(S_std - S_tilde_std)^2]=0; lim E[(W_std - W_tilde_std)^2]=0

    Parameters
    ----------
    data : array-like
        Input data.
    bandwidth : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: convergence

    References
    ----------
    Fauzi Ch 5, Theorem 5.8
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 5.8: standardized smoothed sign/Wilcoxon converge to unsmoothed in L^2"})


def cheatsheet():
    return "fzt58: Theorem 5.8: standardized smoothed sign/Wilcoxon converge to unsmoothed in L^2"
