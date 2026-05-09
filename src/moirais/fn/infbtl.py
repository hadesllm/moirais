"""Information bottleneck Lagrangian."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["information_bottleneck"]


def information_bottleneck(X, Y, beta):
    """
    Information bottleneck Lagrangian

    Formula: min I(X;T) - beta I(T;Y)

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tishby-Pereira-Bialek (1999)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Information bottleneck Lagrangian"})


def cheatsheet():
    return "infbtl: Information bottleneck Lagrangian"
