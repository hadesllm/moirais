# moirais.fn — function file (hadesllm/moirais)
"""Probability Integral Transformation: Y = F_X(X) ~ Uniform(0,1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_pit"]


def gibbons_pit(X, F):
    """
    Probability Integral Transformation: Y = F_X(X) ~ Uniform(0,1)

    Formula: Y = F_X(X) ~ Uniform(0,1) when F_X is continuous

    Parameters
    ----------
    X : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: uniform_sample

    References
    ----------
    Gibbons Theorem 2.5.1
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Probability Integral Transformation: Y = F_X(X) ~ Uniform(0,1)"})


def cheatsheet():
    return "gb251: Probability Integral Transformation: Y = F_X(X) ~ Uniform(0,1)"
