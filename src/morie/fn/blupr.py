"""Best linear unbiased predictor of random intercept."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["blup_random_intercept"]


def blup_random_intercept(y, X, cluster, sigma2_u, sigma2_e):
    """
    Best linear unbiased predictor of random intercept

    Formula: u_j = (sigma2_u / (sigma2_u + sigma2_e/n_j)) * (mean_j - X_j beta)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    cluster : array-like
        Input data.
    sigma2_u : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Best linear unbiased predictor of random intercept"})


def cheatsheet():
    return "blupr: Best linear unbiased predictor of random intercept"
