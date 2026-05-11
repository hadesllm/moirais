# morie.fn — function file (hadesllm/morie)
"""Variance of total number of runs under null hypothesis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_runs_var"]


def gibbons_runs_var(n1, n2):
    """
    Variance of total number of runs under null hypothesis

    Formula: Var(R) = 2*n1*n2*(2*n1*n2 - n1 - n2) / ((n1+n2)^2*(n1+n2-1))

    Parameters
    ----------
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: variance

    References
    ----------
    Gibbons eq 3.2.8
    """
    n1 = np.asarray(n1, dtype=float)
    n = int(n1) if n1.ndim == 0 else len(n1)
    result = float(np.mean(n1))
    se = float(np.std(n1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance of total number of runs under null hypothesis"})


def cheatsheet():
    return "gb32vr: Variance of total number of runs under null hypothesis"
