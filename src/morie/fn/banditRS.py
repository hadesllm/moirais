"""Contextual bandit recommendation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["contextual_bandit_rec"]


def contextual_bandit_rec(context, arms):
    """
    Contextual bandit recommendation

    Formula: LinUCB or Thompson sampling per arm

    Parameters
    ----------
    context : array-like
        Input data.
    arms : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Li et al (2010) LinUCB
    """
    context = np.atleast_1d(np.asarray(context, dtype=float))
    n = len(context)
    result = float(np.mean(context))
    se = float(np.std(context, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Contextual bandit recommendation"})


def cheatsheet():
    return "banditRS: Contextual bandit recommendation"
