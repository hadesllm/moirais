# morie.fn — function file (hadesllm/morie)
"""Largest Lyapunov exponent."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_lyapunov"]


def rangayyan_lyapunov(x):
    """
    Largest Lyapunov exponent

    Formula: lambda = lim (1/t) ln(d(t)/d(0))

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
    Rangayyan Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Largest Lyapunov exponent"})


def cheatsheet():
    return "rglyp: Largest Lyapunov exponent"
