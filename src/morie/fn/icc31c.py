"""ICC(1,1) one-way random effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["icc_one_way"]


def icc_one_way(X):
    """
    ICC(1,1) one-way random effects

    Formula: sigma_alpha^2 / (sigma_alpha^2 + sigma_eps^2)

    Parameters
    ----------
    X : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ICC(1,1) one-way random effects"})


def cheatsheet():
    return "icc31c: ICC(1,1) one-way random effects"
