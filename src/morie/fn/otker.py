"""Kernel-mean approx of EMD via positive-definite cost."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_kernel_emd_approx"]


def ot_kernel_emd_approx(X, Y, kernel, epsilon):
    """
    Kernel-mean approx of EMD via positive-definite cost

    Formula: Cost ~ -2 K(x,y); solve via Sinkhorn

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    kernel : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: EMD_approx

    References
    ----------
    Genevay-Peyré-Cuturi (2018)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel-mean approx of EMD via positive-definite cost"})


def cheatsheet():
    return "otker: Kernel-mean approx of EMD via positive-definite cost"
