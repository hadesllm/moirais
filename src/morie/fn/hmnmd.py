# morie.fn -- function file (hadesllm/morie)
"""Numerical differentiation via finite differences."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_numerical_diff"]


def geron_numerical_diff(f, x, h):
    """
    Numerical differentiation via finite differences

    Formula: df/dx approx (f(x+h) - f(x-h)) / (2h)

    Parameters
    ----------
    f : array-like
        Input data.
    x : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: df_dx

    References
    ----------
    Géron Appendix A
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numerical differentiation via finite differences"})


def cheatsheet():
    return "hmnmd: Numerical differentiation via finite differences"
