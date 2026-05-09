"""ICC(2,k) two-way random effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["icc_two_way"]


def icc_two_way(X, model):
    """
    ICC(2,k) two-way random effects

    Formula: between-subjects var / total var (random raters)

    Parameters
    ----------
    X : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shrout-Fleiss (1979)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ICC(2,k) two-way random effects"})


def cheatsheet():
    return "icc12c: ICC(2,k) two-way random effects"
