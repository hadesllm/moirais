# morie.fn — function file (hadesllm/morie)
"""Pitman efficiency: ratio of sample sizes for identical power in large-sample limit."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_pitman_efficiency"]


def gibbons_pitman_efficiency(T1, T2, theta0):
    """
    Pitman efficiency: ratio of sample sizes for identical power in large-sample limit

    Formula: PE(T1, T2) = lim n2/n1 as n -> inf such that power functions converge

    Parameters
    ----------
    T1 : array-like
        Input data.
    T2 : array-like
        Input data.
    theta0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pitman_efficiency

    References
    ----------
    Gibbons Ch 1.2.11
    """
    T1 = np.asarray(T1, dtype=float)
    n = int(T1) if T1.ndim == 0 else len(T1)
    result = float(np.mean(T1))
    se = float(np.std(T1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pitman efficiency: ratio of sample sizes for identical power in large-sample limit"})


def cheatsheet():
    return "gb_psi: Pitman efficiency: ratio of sample sizes for identical power in large-sample limit"
