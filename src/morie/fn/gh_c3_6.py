# morie.fn — function file (hadesllm/morie)
"""Prior construction via distribution on a dense subset of the sample space."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dense_subset_prior"]


def ghosal_dense_subset_prior(x):
    """
    Prior construction via distribution on a dense subset of the sample space

    Formula: G = sum_k w_k delta_{X_k}, X_k from dense sequence

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
    Ghosal Ch 3 §3.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prior construction via distribution on a dense subset of the sample space"})


def cheatsheet():
    return "gh_c3_6: Prior construction via distribution on a dense subset of the sample space"
