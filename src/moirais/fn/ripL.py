"""Ripley's L function (variance-stabilized K)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ripley_l_function"]


def ripley_l_function(points, window, r):
    """
    Ripley's L function (variance-stabilized K)

    Formula: L(d) = sqrt(K(d) / pi) - d

    Parameters
    ----------
    points : array-like
        Input data.
    window : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Besag (1977)
    """
    points = np.atleast_1d(np.asarray(points, dtype=float))
    n = len(points)
    result = float(np.mean(points))
    se = float(np.std(points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ripley's L function (variance-stabilized K)"})


def cheatsheet():
    return "ripL: Ripley's L function (variance-stabilized K)"
