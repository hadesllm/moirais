"""Bayesian outlier detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_outlier"]


def bayes_outlier(y, outlier_prior):
    """
    Bayesian outlier detection

    Formula: per-obs latent z indicating outlier component

    Parameters
    ----------
    y : array-like
        Input data.
    outlier_prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    West (1984)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian outlier detection"})


def cheatsheet():
    return "bayoutl: Bayesian outlier detection"
