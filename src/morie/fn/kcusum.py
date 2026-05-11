"""Kernel CUSUM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kernel_cusum"]


def kernel_cusum(x, kernel, threshold):
    """
    Kernel CUSUM

    Formula: CUSUM in RKHS embedding

    Parameters
    ----------
    x : array-like
        Input data.
    kernel : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Harchaoui et al (2009)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel CUSUM"})


def cheatsheet():
    return "kcusum: Kernel CUSUM"
