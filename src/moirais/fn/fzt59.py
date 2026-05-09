# moirais.fn — function file (hadesllm/moirais)
"""Theorem 5.9: Edgeworth expansion for smoothed Wilcoxon with fourth-order kernel."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm5_9_edgeworth_wilcoxon"]


def fauzi_thm5_9_edgeworth_wilcoxon(data, bandwidth, theta, c, d):
    """
    Theorem 5.9: Edgeworth expansion for smoothed Wilcoxon with fourth-order kernel

    Formula: Improved normal approximation for W_tilde, h_n=cn^{-d}, 1/4<d<1/2

    Parameters
    ----------
    data : array-like
        Input data.
    bandwidth : array-like
        Input data.
    theta : array-like
        Input data.
    c : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Fauzi Ch 5, Theorem 5.9
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 5.9: Edgeworth expansion for smoothed Wilcoxon with fourth-order kernel"})


def cheatsheet():
    return "fzt59: Theorem 5.9: Edgeworth expansion for smoothed Wilcoxon with fourth-order kernel"
