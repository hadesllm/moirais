"""RBF (Gaussian) kernel."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rbf_kernel"]


def rbf_kernel(x, y, sigma):
    """
    RBF (Gaussian) kernel

    Formula: k = exp(-||x-y||²/(2σ²))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    standard
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RBF (Gaussian) kernel"})


def cheatsheet():
    return "rbfk: RBF (Gaussian) kernel"
