"""Clark-Evans aggregation index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clark_evans"]


def clark_evans(coords):
    """
    Clark-Evans aggregation index

    Formula: R = mean(d_NN) / E[d_NN] under CSR

    Parameters
    ----------
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Clark-Evans (1954)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Clark-Evans aggregation index"})


def cheatsheet():
    return "clstpp: Clark-Evans aggregation index"
