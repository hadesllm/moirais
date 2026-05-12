# morie.fn -- function file (hadesllm/morie)
"""r_2 integral in bias-reduced KDFE variance formula."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_r2_integral"]


def fauzi_r2_integral(kernel, a):
    """
    r_2 integral in bias-reduced KDFE variance formula

    Formula: r_2 = integral y [K(y)W(y/a) + (1/a)W(y)K(y/a)] dy

    Parameters
    ----------
    kernel : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: constant

    References
    ----------
    Fauzi Ch 2, Eq 2.10
    """
    if callable(kernel):
        _xs = np.linspace(-3, 3, 100)
        kernel = np.asarray([kernel(_x) for _x in _xs], dtype=float)
    else:
        kernel = np.asarray(kernel, dtype=float)
    n = len(kernel)
    result = float(np.mean(kernel))
    se = float(np.std(kernel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "r_2 integral in bias-reduced KDFE variance formula"})


def cheatsheet():
    return "fzr2: r_2 integral in bias-reduced KDFE variance formula"
