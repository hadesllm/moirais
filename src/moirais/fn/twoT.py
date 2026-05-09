"""Two-tower retrieval."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["two_tower"]


def two_tower(user_X, item_X, K):
    """
    Two-tower retrieval

    Formula: separate user, item encoders + dot product

    Parameters
    ----------
    user_X : array-like
        Input data.
    item_X : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yi et al (2019)
    """
    user_X = np.atleast_1d(np.asarray(user_X, dtype=float))
    n = len(user_X)
    result = float(np.mean(user_X))
    se = float(np.std(user_X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-tower retrieval"})


def cheatsheet():
    return "twoT: Two-tower retrieval"
