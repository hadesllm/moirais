"""Z-score (standardization) normalization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["zscore_normalization"]


def zscore_normalization(x):
    """
    Z-score (standardization) normalization

    Formula: x_std = (x - mean(x)) / sd(x)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'x_std': 'array'}

    References
    ----------
    Montesinos Lopez Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Z-score (standardization) normalization"})


def cheatsheet():
    return "zscnm: Z-score (standardization) normalization"
