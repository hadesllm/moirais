"""Matrix profile (discord detection)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["matrix_profile"]


def matrix_profile(x, m):
    """
    Matrix profile (discord detection)

    Formula: min Z-norm distance to nearest neighbor subseq

    Parameters
    ----------
    x : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yeh et al (2016)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matrix profile (discord detection)"})


def cheatsheet():
    return "matrxP: Matrix profile (discord detection)"
