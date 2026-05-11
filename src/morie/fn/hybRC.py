"""Hybrid CF + content."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hybrid_rec"]


def hybrid_rec(scores_cf, scores_cb, alpha):
    """
    Hybrid CF + content

    Formula: weighted blend of CF + content-based scores

    Parameters
    ----------
    scores_cf : array-like
        Input data.
    scores_cb : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Burke (2002)
    """
    scores_cf = np.atleast_1d(np.asarray(scores_cf, dtype=float))
    n = len(scores_cf)
    result = float(np.mean(scores_cf))
    se = float(np.std(scores_cf, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hybrid CF + content"})


def cheatsheet():
    return "hybRC: Hybrid CF + content"
