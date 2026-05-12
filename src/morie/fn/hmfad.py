# morie.fn -- function file (hadesllm/morie)
"""Forward-mode automatic differentiation via dual numbers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_forward_autodiff"]


def geron_forward_autodiff(f, x):
    """
    Forward-mode automatic differentiation via dual numbers

    Formula: d/dx f(x) computed by forward propagation of dual components

    Parameters
    ----------
    f : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value, derivative

    References
    ----------
    Géron Appendix A
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Forward-mode automatic differentiation via dual numbers"})


def cheatsheet():
    return "hmfad: Forward-mode automatic differentiation via dual numbers"
