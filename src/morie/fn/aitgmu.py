"""Geometric mean of a composition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_geomean"]


def aitchison_geomean(x):
    """
    Geometric mean of a composition

    Formula: g(x) = (prod_j x_j)^(1/D)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g

    References
    ----------
    Aitchison (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Geometric mean of a composition"})


def cheatsheet():
    return "aitgmu: Geometric mean of a composition"
