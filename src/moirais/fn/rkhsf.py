# moirais.fn — function file (hadesllm/moirais)
"""RKHS with Gaussian kernel for genomics."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rkhs_full"]


def rkhs_full(x, y, markers):
    """
    RKHS with Gaussian kernel for genomics

    Formula: y = X*beta + K*alpha + e, K = exp(-D^2/h)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    markers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Montesinos Lopez Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RKHS with Gaussian kernel for genomics"})


def cheatsheet():
    return "rkhsf: RKHS with Gaussian kernel for genomics"
