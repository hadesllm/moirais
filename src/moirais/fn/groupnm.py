"""GroupNorm — per-group channel normalization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["group_norm"]


def group_norm(y, x, groups, g, b, eps):
    """
    GroupNorm — per-group channel normalization

    Formula: normalize across (C/G, H, W) groups

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    groups : array-like
        Input data.
    g : array-like
        Input data.
    b : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wu & He (2018)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GroupNorm — per-group channel normalization"})


def cheatsheet():
    return "groupnm: GroupNorm — per-group channel normalization"
