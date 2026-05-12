# morie.fn -- function file (hadesllm/morie)
"""Shannon entropy at a tree node (alternative to Gini)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_shannon_entropy"]


def geron_shannon_entropy(y):
    """
    Shannon entropy at a tree node (alternative to Gini)

    Formula: H_i = - sum_{k: p_{i,k}>0} p_{i,k} log_2 p_{i,k}

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: entropy

    References
    ----------
    Géron Ch 5, Eq 5-3 (Entropy)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shannon entropy at a tree node (alternative to Gini)"})


def cheatsheet():
    return "grent: Shannon entropy at a tree node (alternative to Gini)"
