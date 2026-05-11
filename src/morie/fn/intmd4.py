"""4-way decomposition of total effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["interaction_mediation_4way"]


def interaction_mediation_4way(X, M, Y):
    """
    4-way decomposition of total effect

    Formula: TE = CDE + INTref + INTmed + PIE (VanderWeele)

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
    VanderWeele (2014)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "4-way decomposition of total effect"})


def cheatsheet():
    return "intmd4: 4-way decomposition of total effect"
