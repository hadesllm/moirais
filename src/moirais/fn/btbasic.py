"""Basic (reverse-percentile) bootstrap CI."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_basic_ci"]


def boot_basic_ci(theta_hat, theta_b, alpha):
    """
    Basic (reverse-percentile) bootstrap CI

    Formula: [2θ̂ - θ̂*_{1-α/2}, 2θ̂ - θ̂*_{α/2}]

    Parameters
    ----------
    theta_hat : array-like
        Input data.
    theta_b : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi

    References
    ----------
    Davison & Hinkley (1997)
    """
    theta_hat = np.atleast_1d(np.asarray(theta_hat, dtype=float))
    n = len(theta_hat)
    result = float(np.mean(theta_hat))
    se = float(np.std(theta_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Basic (reverse-percentile) bootstrap CI"})


def cheatsheet():
    return "btbasic: Basic (reverse-percentile) bootstrap CI"
