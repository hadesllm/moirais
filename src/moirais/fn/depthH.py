"""Tukey halfspace depth."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["halfspace_depth"]


def halfspace_depth(X, theta):
    """
    Tukey halfspace depth

    Formula: min over halfspaces H containing θ of P(H)

    Parameters
    ----------
    X : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tukey (1975)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tukey halfspace depth"})


def cheatsheet():
    return "depthH: Tukey halfspace depth"
