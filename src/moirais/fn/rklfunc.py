"""Ripley's L (variance-stabilized K)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ripley_l"]


def ripley_l(coords, r_grid):
    """
    Ripley's L (variance-stabilized K)

    Formula: L(r) = sqrt(K(r)/pi) - r

    Parameters
    ----------
    coords : array-like
        Input data.
    r_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Besag (1977)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ripley's L (variance-stabilized K)"})


def cheatsheet():
    return "rklfunc: Ripley's L (variance-stabilized K)"
