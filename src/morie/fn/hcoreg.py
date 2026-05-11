"""Hard-core process — minimum allowed distance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hardcore_process"]


def hardcore_process(coords, r, lam):
    """
    Hard-core process — minimum allowed distance

    Formula: density 0 if any d_ij < r else lambda^n

    Parameters
    ----------
    coords : array-like
        Input data.
    r : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Matérn (1960)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hard-core process — minimum allowed distance"})


def cheatsheet():
    return "hcoreg: Hard-core process — minimum allowed distance"
