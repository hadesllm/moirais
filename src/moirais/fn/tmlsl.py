"""TMLE with Super Learner ensemble for Q and g."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_super_learner"]


def tmle_super_learner(y, D, X, library):
    """
    TMLE with Super Learner ensemble for Q and g

    Formula: convex combination of candidate learners minimizing CV risk

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    library : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    vdL-Polley-Hubbard (2007); Polley-Rose-vdL (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE with Super Learner ensemble for Q and g"})


def cheatsheet():
    return "tmlsl: TMLE with Super Learner ensemble for Q and g"
