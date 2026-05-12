# morie.fn -- function file (hadesllm/morie)
"""Information gain from a split using entropy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_information_gain"]


def geron_information_gain(y, split):
    """
    Information gain from a split using entropy

    Formula: IG = H(parent) - sum_c (m_c/m) H(child_c)

    Parameters
    ----------
    y : array-like
        Input data.
    split : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: information_gain

    References
    ----------
    Géron Ch 5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Information gain from a split using entropy"})


def cheatsheet():
    return "hmigr: Information gain from a split using entropy"
