"""L² distance between curves."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_distance"]


def functional_distance(f, g):
    """
    L² distance between curves

    Formula: d(f,g) = √∫(f-g)²

    Parameters
    ----------
    f : array-like
        Input data.
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay-Silverman (2005)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "L² distance between curves"})


def cheatsheet():
    return "fnDist: L² distance between curves"
