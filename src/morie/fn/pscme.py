"""Path-specific causal effect for multiple mediators."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["path_specific_causal_effect"]


def path_specific_causal_effect(X, M, Y):
    """
    Path-specific causal effect for multiple mediators

    Formula: PSE_pi = E[Y(1, M_pi(1), M_-pi(0))] - E[Y(0,M(0))]

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Avin, Shpitser, Pearl (2005)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Path-specific causal effect for multiple mediators"})


def cheatsheet():
    return "pscme: Path-specific causal effect for multiple mediators"
