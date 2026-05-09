# moirais.fn — function file (hadesllm/moirais)
"""Vanilla gradient descent optimizer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gradient_descent_vanilla"]


def gradient_descent_vanilla(x, y):
    """
    Vanilla gradient descent optimizer

    Formula: theta = theta - alpha * grad(J)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vanilla gradient descent optimizer"})


def cheatsheet():
    return "grdds: Vanilla gradient descent optimizer"
