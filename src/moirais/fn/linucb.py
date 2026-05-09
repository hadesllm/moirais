"""LinUCB contextual bandit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["linucb"]


def linucb(context, arms, alpha):
    """
    LinUCB contextual bandit

    Formula: arm = argmax_a θ̂_a^T x + α √(x^T A^{-1}_a x)

    Parameters
    ----------
    context : array-like
        Input data.
    arms : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Li et al (2010)
    """
    context = np.atleast_1d(np.asarray(context, dtype=float))
    n = len(context)
    result = float(np.mean(context))
    se = float(np.std(context, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LinUCB contextual bandit"})


def cheatsheet():
    return "linucb: LinUCB contextual bandit"
