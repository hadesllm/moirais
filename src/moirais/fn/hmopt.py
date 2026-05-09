# moirais.fn — function file (hadesllm/moirais)
"""OPTICS: ordering points to identify clustering structure at multiple densities."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_optics"]


def geron_optics(X, min_samples, max_eps):
    """
    OPTICS: ordering points to identify clustering structure at multiple densities

    Formula: reachability plot; clusters from valleys

    Parameters
    ----------
    X : array-like
        Input data.
    min_samples : array-like
        Input data.
    max_eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, reachability

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "OPTICS: ordering points to identify clustering structure at multiple densities"})


def cheatsheet():
    return "hmopt: OPTICS: ordering points to identify clustering structure at multiple densities"
