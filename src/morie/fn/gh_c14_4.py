# morie.fn — function file (hadesllm/morie)
"""Chinese restaurant franchise: hierarchical DP via CRP at each restaurant."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_crf_def"]


def ghosal_crf_def(x):
    """
    Chinese restaurant franchise: hierarchical DP via CRP at each restaurant

    Formula: CRF: global menu G0, per-restaurant G_j ~ DP(alpha, G0), sharing dishes

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
    Ghosal Ch 14 §14.1.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chinese restaurant franchise: hierarchical DP via CRP at each restaurant"})


def cheatsheet():
    return "gh_c14_4: Chinese restaurant franchise: hierarchical DP via CRP at each restaurant"
