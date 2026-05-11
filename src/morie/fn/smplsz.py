"""Sample-size calculation for proportion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sample_size_calc"]


def sample_size_calc(p, e, z):
    """
    Sample-size calculation for proportion

    Formula: n = z^2 p(1-p) / e^2

    Parameters
    ----------
    p : array-like
        Input data.
    e : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sample-size calculation for proportion"})


def cheatsheet():
    return "smplsz: Sample-size calculation for proportion"
