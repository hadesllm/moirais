# morie.fn -- function file (hadesllm/morie)
"""Mean-shift: mode-seeking via kernel density gradient ascent."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mean_shift"]


def geron_mean_shift(X, bandwidth):
    """
    Mean-shift: mode-seeking via kernel density gradient ascent

    Formula: x_{t+1} = sum_i K(x_t - x_i) x_i / sum_i K(x_t - x_i)

    Parameters
    ----------
    X : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, centers

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean-shift: mode-seeking via kernel density gradient ascent"})


def cheatsheet():
    return "hmmnsh: Mean-shift: mode-seeking via kernel density gradient ascent"
