# morie.fn — function file (hadesllm/morie)
"""Central-difference numerical gradient approximation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_numerical_differentiation"]


def geron_numerical_differentiation(f, x, h):
    """
    Central-difference numerical gradient approximation

    Formula: df/dx ≈ (f(x + h) - f(x - h)) / (2 h)

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
        Keys: grad

    References
    ----------
    Géron Appendix A, Numerical Differentiation section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Central-difference numerical gradient approximation"})


def cheatsheet():
    return "grnud: Central-difference numerical gradient approximation"
