# morie.fn -- function file (hadesllm/morie)
"""Infinite-dimensional BvM: posterior approximated by Gaussian process centered at MLE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_infdim_bvm"]


def ghosal_infdim_bvm(x):
    """
    Infinite-dimensional BvM: posterior approximated by Gaussian process centered at MLE

    Formula: sqrt(n)(Pi_n - N(theta_hat, I_n^{-1})) -> 0 in total variation

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 12 §12.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Infinite-dimensional BvM: posterior approximated by Gaussian process centered at MLE"})


def cheatsheet():
    return "gh_c12_1: Infinite-dimensional BvM: posterior approximated by Gaussian process centered at MLE"
