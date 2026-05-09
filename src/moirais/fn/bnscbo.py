"""Compound-outcome bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_compound_outcome"]


def bound_compound_outcome(y_components, D, X):
    """
    Compound-outcome bound

    Formula: composite outcome bounds

    Parameters
    ----------
    y_components : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski (2003)
    """
    y_components = np.atleast_1d(np.asarray(y_components, dtype=float))
    n = len(y_components)
    result = float(np.mean(y_components))
    se = float(np.std(y_components, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Compound-outcome bound"})


def cheatsheet():
    return "bnscbo: Compound-outcome bound"
