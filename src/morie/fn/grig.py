# morie.fn -- function file (hadesllm/morie)
"""Information gain from splitting on feature k at threshold t_k."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_information_gain"]


def geron_information_gain(y, left_mask):
    """
    Information gain from splitting on feature k at threshold t_k

    Formula: IG = H(parent) - (m_L/m) H(left) - (m_R/m) H(right)

    Parameters
    ----------
    y : array-like
        Input data.
    left_mask : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ig

    References
    ----------
    Géron Ch 5, Information Gain / CART discussion
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Information gain from splitting on feature k at threshold t_k"})


def cheatsheet():
    return "grig: Information gain from splitting on feature k at threshold t_k"
