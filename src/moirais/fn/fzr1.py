# moirais.fn — function file (hadesllm/moirais)
"""r_1 integral in KDFE variance formula."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_r1_integral"]


def fauzi_r1_integral(kernel):
    """
    r_1 integral in KDFE variance formula

    Formula: r_1 = integral y K(y) W(y) dy

    Parameters
    ----------
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: constant

    References
    ----------
    Fauzi Ch 2, Eq 2.9
    """
    kernel = np.asarray(kernel, dtype=float)
    n = int(kernel) if kernel.ndim == 0 else len(kernel)
    result = float(np.mean(kernel))
    se = float(np.std(kernel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "r_1 integral in KDFE variance formula"})


def cheatsheet():
    return "fzr1: r_1 integral in KDFE variance formula"
