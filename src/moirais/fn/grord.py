# moirais.fn — function file (hadesllm/moirais)
"""Ordinal encoding: map K ordered categories to {0,1,...,K-1}."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ordinal_encoding"]


def geron_ordinal_encoding(categories):
    """
    Ordinal encoding: map K ordered categories to {0,1,...,K-1}

    Formula: enc(c_k) = k for k in 0..K-1

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
    Géron Ch 2, Ordinal Encoding section
    """
    categories = np.asarray(categories, dtype=float)
    n = int(categories) if categories.ndim == 0 else len(categories)
    result = float(np.mean(categories))
    se = float(np.std(categories, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ordinal encoding: map K ordered categories to {0,1,...,K-1}"})


def cheatsheet():
    return "grord: Ordinal encoding: map K ordered categories to {0,1,...,K-1}"
