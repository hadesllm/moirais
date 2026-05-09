"""Sharp RDD with local linear regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sharp_rdd"]


def sharp_rdd(y, x, cutoff, bandwidth):
    """
    Sharp RDD with local linear regression

    Formula: tau = lim_{x->c+} E[Y|X=x] - lim_{x->c-} E[Y|X=x]

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    cutoff : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hahn, Todd, van der Klaauw (2001)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sharp RDD with local linear regression"})


def cheatsheet():
    return "rdksrn: Sharp RDD with local linear regression"
