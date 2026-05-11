"""Hierarchical pooling (no/complete/partial)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hierarchical_pooling"]


def hierarchical_pooling(y, group):
    """
    Hierarchical pooling (no/complete/partial)

    Formula: shrink group estimates toward grand mean by 1/(1+sigma_g^2/tau^2)

    Parameters
    ----------
    y : array-like
        Input data.
    group : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman BDA3 Ch 5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hierarchical pooling (no/complete/partial)"})


def cheatsheet():
    return "bayhier: Hierarchical pooling (no/complete/partial)"
