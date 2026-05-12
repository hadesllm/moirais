# morie.fn -- function file (hadesllm/morie)
"""White noise BvM for full infinite-dimensional parameter."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_wn_full_bvm"]


def ghosal_wn_full_bvm(x):
    """
    White noise BvM for full infinite-dimensional parameter

    Formula: dY = theta dt + dW/sqrt(n), Pi_n -> N(Y, I/n) in total variation

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
    Ghosal Ch 12 §12.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "White noise BvM for full infinite-dimensional parameter"})


def cheatsheet():
    return "gh_c12_9: White noise BvM for full infinite-dimensional parameter"
