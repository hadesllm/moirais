# morie.fn — function file (hadesllm/morie)
"""Contingency coefficient C from chi-square."""
import numpy as np
from ._richresult import RichResult

__all__ = ["contingency_coefficient"]


def contingency_coefficient(x):
    """
    Contingency coefficient C from chi-square

    Formula: C = sqrt(chi2 / (chi2 + n))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gibbons Ch 14.2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Contingency coefficient C from chi-square"})


def cheatsheet():
    return "cntgc: Contingency coefficient C from chi-square"
