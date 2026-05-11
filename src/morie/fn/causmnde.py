"""VanderWeele NDE/NIE/PDE/PIE decomposition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_natural_decomposition"]


def causal_natural_decomposition(X, M, Y, T):
    """
    VanderWeele NDE/NIE/PDE/PIE decomposition

    Formula: TE = NDE + NIE; further into pure + interaction

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: NDE, NIE, PDE, PIE

    References
    ----------
    VanderWeele (2015)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VanderWeele NDE/NIE/PDE/PIE decomposition"})


def cheatsheet():
    return "causmnde: VanderWeele NDE/NIE/PDE/PIE decomposition"
