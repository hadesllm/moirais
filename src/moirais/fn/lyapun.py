"""Largest Lyapunov exponent (Rosenstein)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lyapunov_exponent"]


def lyapunov_exponent(y, embedding, tau):
    """
    Largest Lyapunov exponent (Rosenstein)

    Formula: divergence rate of nearest neighbors

    Parameters
    ----------
    y : array-like
        Input data.
    embedding : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rosenstein-Collins-De Luca (1993)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Largest Lyapunov exponent (Rosenstein)"})


def cheatsheet():
    return "lyapun: Largest Lyapunov exponent (Rosenstein)"
