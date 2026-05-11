# morie.fn — function file (hadesllm/morie)
"""One-hot encoding of categorical feature into indicator columns."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_one_hot_encoding"]


def geron_one_hot_encoding(categories):
    """
    One-hot encoding of categorical feature into indicator columns

    Formula: e_k(c) = 1 if c==k else 0, for k in unique(C)

    Parameters
    ----------
    categories : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: encoded

    References
    ----------
    Géron Ch 2, One-hot Encoding section
    """
    categories = np.asarray(categories, dtype=float)
    n = int(categories) if categories.ndim == 0 else len(categories)
    result = float(np.mean(categories))
    se = float(np.std(categories, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "One-hot encoding of categorical feature into indicator columns"})


def cheatsheet():
    return "grohe: One-hot encoding of categorical feature into indicator columns"
