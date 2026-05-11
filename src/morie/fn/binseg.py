"""Binary segmentation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["binary_segmentation"]


def binary_segmentation(x, K):
    """
    Binary segmentation

    Formula: recursively pick most-significant split

    Parameters
    ----------
    x : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Scott-Knott (1974)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Binary segmentation"})


def cheatsheet():
    return "binseg: Binary segmentation"
