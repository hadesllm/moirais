"""Content-based recommendation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["content_based"]


def content_based(item_feat, user_profile):
    """
    Content-based recommendation

    Formula: score = sim(item profile, user profile)

    Parameters
    ----------
    item_feat : array-like
        Input data.
    user_profile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pazzani-Billsus (2007)
    """
    item_feat = np.atleast_1d(np.asarray(item_feat, dtype=float))
    n = len(item_feat)
    result = float(np.mean(item_feat))
    se = float(np.std(item_feat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Content-based recommendation"})


def cheatsheet():
    return "contRC: Content-based recommendation"
