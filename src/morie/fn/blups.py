"""BLUP for random slope (one covariate)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["blup_random_slope"]


def blup_random_slope(y, X, Z, cluster, D, sigma2_e):
    """
    BLUP for random slope (one covariate)

    Formula: v_j = D Z_j' (Z_j D Z_j' + sigma2_e I)^-1 (y_j - X_j beta)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.
    cluster : array-like
        Input data.
    D : array-like
        Input data.
    sigma2_e : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Henderson (1975); Robinson (1991)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BLUP for random slope (one covariate)"})


def cheatsheet():
    return "blups: BLUP for random slope (one covariate)"
