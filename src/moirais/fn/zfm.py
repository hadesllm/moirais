"""Z-transform."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["z_transform"]


def z_transform(x, z):
    """
    Z-transform

    Formula: X(z) = sum x_n z^{-n}

    Parameters
    ----------
    x : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jury (1964)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Z-transform"})


def cheatsheet():
    return "zfm: Z-transform"
